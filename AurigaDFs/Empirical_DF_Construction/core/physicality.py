"""
The physicality conditions of Table 1, tested on a fitted form over the whole of (0, 1):

    low-energy        f -> 0 as eps_hat -> 0       (log10 f -> -oo)
    monotonicity      df/deps_hat > 0 on (0, 1)
    well-definedness  f finite and real on (0, 1)

The fits only see eps_hat up to ~0.999, so the curve is generated from the expression on a
grid reaching 1e-300 from both ends. Float64 cannot write 1 - 1e-20, so the upper half is
swept in t = 1 - eps_hat instead.

Monotonicity and well-definedness are swept numerically. A sweep can refute but never
prove, so their verdicts are REFUTED or NOT_REFUTED. Each counterexample is re-checked with
mpmath at a precision that grows with the abscissa before it counts.

The low-energy limit is decided symbolically by sympy (Gruntz), since a sweep cannot tell a
slow divergence like -log(-log(eps_hat)) from a finite limit. It can return HOLDS. The limit
is capped by a timeout, and a sum is retried term by term.

Every phase is time-capped. A phase that runs out is UNDECIDED: neither a pass nor a
refutation, and reported separately (`Physicality.status`).
"""
from __future__ import annotations

import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

import mpmath as mp
import numpy as np
import sympy as sp

from .df_transforms import EPS, SYMPY_TRANSFORMS

T = sp.Symbol("t", positive=True)   # t = 1 - eps_hat, the upper half's variable

DECADES = 300           # the sweep reaches 10**-DECADES from each end
PER_DECADE = 64         # log-spaced abscissae per decade
N_MIDDLE = 20_000       # linear abscissae across the middle of each half
CONFIRM_DPS = 50        # minimum digits a counterexample is re-checked at
LADDER_DECADES = 30     # decades of the low-energy ladder (evidence only)
LIMIT_TIMEOUT = 300     # seconds for the symbolic low-energy limit
TEST_TIMEOUT = 300      # seconds for each numeric phase
IM_TOL = 1e-9
POLE_MAG = 1e6          # |log10 f| above which a sign flip is a pole, not a zero


class Verdict(str, Enum):
    REFUTED = "refuted"
    NOT_REFUTED = "not_refuted"   # no counterexample found; not a proof
    HOLDS = "holds"               # proved (only the symbolic limit can say this)
    UNDECIDED = "undecided"       # sympy gave no answer, or the timeout ran out


@dataclass
class Side:
    """One half of the sweep. `v` is eps_hat on the lower half and t = 1 - eps_hat on the
    upper; it is never converted to eps_hat, which would lose the region near 1."""
    name: str
    v: np.ndarray
    y: np.ndarray                 # log10 f, NaN where undefined
    dy: np.ndarray                # d(log10 f)/d(eps_hat), NaN where undefined
    expr: sp.Expr = None          # log10 f and its eps_hat-derivative, in this side's variable
    dexpr: sp.Expr = None
    var: sp.Symbol = None

    def eps(self, i: int) -> str:
        return f"{self.v[i]:.12g}" if self.name == "lower" else f"1 - {self.v[i]:.12g}"


@dataclass
class Physicality:
    low_energy: Verdict
    monotone: Verdict
    well_defined: Verdict
    ladder: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    sides: list = field(default_factory=list)

    @property
    def refuted(self) -> bool:
        return Verdict.REFUTED in (self.monotone, self.well_defined, self.low_energy)

    @property
    def timed_out(self) -> bool:
        return bool(self.detail.get("timed_out_on"))

    @property
    def survives(self) -> bool:
        """Not refuted, and every condition was actually tested."""
        return not self.refuted and not self.timed_out

    @property
    def status(self) -> str:
        """'refuted', 'undecided' (a phase timed out) or 'survives'. Refuted wins."""
        if self.refuted:
            return "refuted"
        return "undecided" if self.timed_out else "survives"


# --------------------------------------------------------------------------------------
# generating the curve
# --------------------------------------------------------------------------------------

def half_grid(decades: int = DECADES, per_decade: int = PER_DECADE,
              n_middle: int = N_MIDDLE) -> np.ndarray:
    """From 10**-decades up to 1/2, log-spaced then linear. Used for both halves."""
    log = np.logspace(-decades, np.log10(0.5), decades * per_decade)
    lin = np.linspace(1e-3, 0.5, n_middle)
    g = np.unique(np.concatenate([log, lin]))
    return g[(g > 0) & (g <= 0.5)]


def as_expr(equation: str, transform: str | None = None) -> sp.Expr:
    """`equation_eps`, or `equation_u` with the transform it was fitted under, in eps_hat."""
    e = sp.sympify(str(equation).replace("^", "**"),
                   locals={"eps_hat": EPS, "u": sp.Symbol("u")})
    if transform is not None and transform != "identity":
        if transform not in SYMPY_TRANSFORMS:
            raise KeyError(f"unknown transform {transform!r}")
        e = e.subs(sp.Symbol("u"), SYMPY_TRANSFORMS[transform])
    return e.subs(sp.Symbol("u"), EPS)


def _evaluate(expr: sp.Expr, var: sp.Symbol, x: np.ndarray) -> np.ndarray:
    """Vectorised values, NaN wherever the result is not a finite real. Evaluated in complex
    so that e.g. a log of a negative number shows up as off-domain instead of raising."""
    try:
        f = sp.lambdify(var, expr, "numpy")
        with np.errstate(all="ignore"):
            y = np.asarray(f(x.astype(complex)), dtype=complex)
    except Exception:
        return np.full(x.shape, np.nan)
    y = np.broadcast_to(y, x.shape).copy()
    out = np.where(np.abs(y.imag) < IM_TOL, y.real, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def generate(e: sp.Expr, grid: np.ndarray | None = None) -> list[Side]:
    """Both halves of (0, 1). The derivative is with respect to eps_hat on both sides."""
    g = half_grid() if grid is None else grid
    d = sp.diff(e, EPS)
    e_up, d_up = e.subs(EPS, 1 - T), d.subs(EPS, 1 - T)
    return [Side("lower", g, _evaluate(e, EPS, g), _evaluate(d, EPS, g), e, d, EPS),
            Side("upper", g, _evaluate(e_up, T, g), _evaluate(d_up, T, g), e_up, d_up, T)]


# --------------------------------------------------------------------------------------
# confirming a counterexample at high precision
# --------------------------------------------------------------------------------------

def dps_for(x: float) -> int:
    """Digits for a re-check at abscissa x. Grows with -log10 x, because the expression still
    contains 1 - t: a fixed 50 digits once 'confirmed' a float64 artefact at t = 1e-300."""
    if not np.isfinite(x) or x <= 0:
        return CONFIRM_DPS
    return max(CONFIRM_DPS, int(-np.log10(x)) + 25)


def _mp_value(expr: sp.Expr, var: sp.Symbol, x: float):
    """One high-precision value, or None where the expression is undefined."""
    with mp.workdps(dps_for(x)):
        try:
            v = sp.lambdify(var, expr, "mpmath")(mp.mpf(float(x)))
        except Exception:
            return None
        if isinstance(v, mp.mpc):
            if mp.fabs(v.imag) > mp.mpf(10) ** (-mp.mp.dps // 2):
                return None                        # complex: off the domain
            v = v.real
        return v if mp.isfinite(v) else None


def _confirm(expr: sp.Expr, var: sp.Symbol, x: float, test) -> bool:
    """Whether `test` still fails at x at high precision (undefined there counts as failing)."""
    v = _mp_value(expr, var, x)
    if v is None:
        return True
    try:
        return bool(test(v))
    except Exception:
        return False


def _pair_falls(side: Side, i: int) -> bool:
    """Whether values i and i+1 strictly fail to increase in eps_hat. Strict on purpose:
    values equal at any precision say nothing (exp(-1e298) differences); the derivative
    check catches a truly flat f."""
    a, b = (_mp_value(side.expr, side.var, side.v[j]) for j in (i, i + 1))
    if a is None or b is None:
        return False
    return (b < a) if side.name == "lower" else (b > a)


# --------------------------------------------------------------------------------------
# the numeric conditions
# --------------------------------------------------------------------------------------

def check_well_defined(sides: list[Side]) -> tuple[Verdict, str]:
    """No undefined abscissa, and no pole between two (a sign flip at |log10 f| > POLE_MAG)."""
    for side in sides:
        bad = np.flatnonzero(~np.isfinite(side.y))
        # check every candidate: NaNs near t = 1e-300 are usually float64 artefacts
        hit = next((int(k) for k in bad if _mp_value(side.expr, side.var, side.v[k]) is None), None)
        if hit is not None:
            return Verdict.REFUTED, (f"undefined at eps_hat = {side.eps(hit)}; "
                                     f"{bad.size} of {side.y.size} abscissae undefined "
                                     f"on the {side.name} half")
    for side in sides:
        y = side.y
        ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
        flip = ok & (np.sign(y[:-1]) != np.sign(y[1:]))
        big = np.minimum(np.abs(y[:-1]), np.abs(y[1:])) > POLE_MAG
        hit = [k for k in np.flatnonzero(flip & big)
               if _confirm(side.expr, side.var, side.v[k], lambda z: mp.fabs(z) > POLE_MAG)]
        if hit:
            i = int(hit[0])
            return Verdict.REFUTED, (
                f"pole between eps_hat = {side.eps(i)} and {side.eps(i + 1)}: "
                f"log10 f passes from {y[i]:.4g} to {y[i + 1]:.4g} across one interval")
    return Verdict.NOT_REFUTED, (f"defined at all {sum(s.y.size for s in sides)} abscissae, "
                                 f"no pole found")


def check_monotone(sides: list[Side]) -> tuple[Verdict, str]:
    """df/deps_hat > 0: by the derivative, then by consecutive values (which must fall with t
    on the upper half)."""
    for side in sides:
        bad = np.flatnonzero(np.isfinite(side.dy) & (side.dy <= 0))
        if bad.size and _confirm(side.dexpr, side.var, side.v[bad[0]], lambda z: z <= 0):
            i = int(bad[0])
            return Verdict.REFUTED, (
                f"dy/deps_hat = {side.dy[i]:.6g} <= 0 at eps_hat = {side.eps(i)} "
                f"({bad.size} such abscissae on the {side.name} half)")
    for side in sides:
        y = side.y
        ok = np.isfinite(y[:-1]) & np.isfinite(y[1:])
        falls = (y[1:] <= y[:-1]) if side.name == "lower" else (y[1:] >= y[:-1])
        hit = [int(k) for k in np.flatnonzero(ok & falls) if _pair_falls(side, int(k))]
        if hit:
            return Verdict.REFUTED, (
                f"generated values are not increasing in eps_hat across "
                f"{side.eps(hit[0])} ({len(hit)} such consecutive pairs, {side.name} half)")
    return Verdict.NOT_REFUTED, (f"increasing at all {sum(s.y.size for s in sides)} "
                                 f"abscissae swept")


def low_energy_ladder(e: sp.Expr, decades: int = LADDER_DECADES) -> list[tuple[str, str]]:
    """log10 f at eps_hat = 1e-1 ... 1e-decades. Evidence for a reader, never a verdict."""
    x = np.array([10.0 ** -k for k in range(1, decades + 1)])
    y = _evaluate(e, EPS, x)
    return [(f"1e-{k + 1}", "undefined" if not np.isfinite(v) else f"{v:.10g}")
            for k, v in enumerate(y)]


# --------------------------------------------------------------------------------------
# time caps and the symbolic low-energy limit
# --------------------------------------------------------------------------------------

class _LimitTimeout(BaseException):
    """Not an Exception, so sympy's broad `except Exception` cannot swallow it."""


def _alarm(signum, frame):
    raise _LimitTimeout()


@contextmanager
def _capped(seconds: int):
    """Raise _LimitTimeout after `seconds`. SIGALRM only works on the main thread (also a
    ProcessPoolExecutor worker's), so elsewhere this runs uncapped."""
    armed, old = False, None
    if seconds and threading.current_thread() is threading.main_thread():
        try:
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(max(1, int(seconds)))
            armed = True
        except ValueError:                          # no SIGALRM on this platform
            pass
    try:
        yield
    finally:
        if armed:
            signal.alarm(0)
            if old is not None:
                signal.signal(signal.SIGALRM, old)


def _decided(v) -> bool:
    return v is not None and (v in (sp.oo, -sp.oo) or v.is_finite)


def _limit_once(e: sp.Expr, timeout: int):
    """lim_{eps_hat -> 0+} e, or None if sympy fails or runs out of time."""
    try:
        with _capped(timeout):
            return sp.limit(e, EPS, 0, "+")
    except (_LimitTimeout, Exception):
        return None


def _termwise_limit(e: sp.Expr, timeout: int):
    """The limit of a sum from the limits of its terms, or None. Every term must decide,
    and +oo with -oo is indeterminate. (A sum that ran >300 s whole took 0.2 s termwise.)"""
    terms = sp.Add.make_args(e)
    if len(terms) < 2:
        return None
    deadline = time.monotonic() + timeout
    lims = []
    for t in terms:
        left = deadline - time.monotonic()
        if left <= 0:
            return None
        v = _limit_once(t, int(left) + 1)
        if not _decided(v):
            return None
        lims.append(v)
    infs = {v for v in lims if v in (sp.oo, -sp.oo)}
    if len(infs) > 1:
        return None
    return infs.pop() if infs else sp.Add(*lims)


def low_energy_limit(e: sp.Expr, timeout: int = LIMIT_TIMEOUT) -> tuple[Verdict, str]:
    """-oo HOLDS; finite or +oo is REFUTED; anything else is UNDECIDED. Two thirds of
    `timeout` go to the whole expression, the rest to the termwise fallback."""
    whole = max(1, (2 * timeout) // 3)
    lim, how = _limit_once(e, whole), ""
    if not _decided(lim):
        lim2 = _termwise_limit(e, max(1, timeout - whole))
        if lim2 is not None:
            lim, how = lim2, " (summed from the limits of its terms)"
    if lim is None:
        return Verdict.UNDECIDED, f"not decided within {timeout}s"
    if lim == -sp.oo:
        return Verdict.HOLDS, "-oo" + how
    if lim == sp.oo or lim.is_finite:
        return Verdict.REFUTED, f"tends to {sp.sstr(lim)}, not to -oo{how}"
    return Verdict.UNDECIDED, f"sympy returned {sp.sstr(lim)}"   # e.g. an unevaluated Limit


# --------------------------------------------------------------------------------------

def check(equation: str, transform: str | None = None,
          grid: np.ndarray | None = None, keep_sides: bool = False,
          limit_timeout: int = LIMIT_TIMEOUT,
          test_timeout: int = TEST_TIMEOUT) -> Physicality:
    """All three conditions for one expression (`equation_eps`, or `equation_u` + `transform`).

    Each numeric phase (sides, well-definedness, monotonicity, ladder) gets `test_timeout`
    and the limit gets `limit_timeout`. Phases that run out are listed in
    detail['timed_out_on'] and make the status 'undecided'.
    """
    try:
        e = as_expr(equation, transform)
    except Exception as exc:
        return Physicality(Verdict.UNDECIDED, Verdict.REFUTED, Verdict.REFUTED,
                           detail={"parse": f"does not parse: {exc}"})

    timed_out = []

    def run(name, fn, *args):
        try:
            with _capped(test_timeout):
                return fn(*args)
        except _LimitTimeout:
            timed_out.append(name)
            return None

    expired = f"not decided within {test_timeout}s"
    sides = run("generate", generate, e, grid)
    if sides is None:
        well = mono = (Verdict.UNDECIDED, f"sides not built within {test_timeout}s")
    else:
        well = run("well_defined", check_well_defined, sides) or (Verdict.UNDECIDED, expired)
        mono = run("monotone", check_monotone, sides) or (Verdict.UNDECIDED, expired)

    low, ld = low_energy_limit(e, timeout=limit_timeout)
    if low is Verdict.UNDECIDED and ld.endswith(f"within {limit_timeout}s"):
        timed_out.append("low_energy")

    ladder = run("ladder", low_energy_ladder, e) or []

    detail = {"n_abscissae": int(sum(s.y.size for s in sides)) if sides else 0,
              "reach": f"1e-{DECADES} to 1 - 1e-{DECADES}" if sides else None,
              "well_defined": well[1], "monotone": mono[1],
              "low_energy": ld,
              "timed_out_on": ",".join(timed_out) if timed_out else None,
              "low_energy_note": "decided by sympy's limit, not by the ladder: a sweep "
                                 "cannot tell a slow divergence from a finite limit"}
    return Physicality(low, mono[0], well[0], ladder=ladder,
                       detail=detail, sides=sides if (keep_sides and sides) else [])
