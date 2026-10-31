"""
Compose a fitted expression back onto eps_hat, and find its simplest exact form.

PySR fits log10 f = g(u) with u = T(eps_hat). The change of variable is part of the
model, not a preprocessing step, so the object to report and to score for complexity is
the composition g(T(eps_hat)). Substituting is mechanical; the work is afterwards.

Composition leaves expressions in forms that are far from their simplest, and the
distortion runs both ways. `exp(u)` under logit is a three-node pile of logs and
exponentials until sympy cancels it to eps_hat/(1 - eps_hat); meanwhile logcombine turns
the composed `(u - 2.13)*1.026` into a single log of a product of two non-integer powers,
which is *larger* than what it started from. So no one simplification is right. Every
candidate rewrite is tried, each is checked numerically on the fitted grid, and the
cheapest survivor wins -- `best_form` is a minimisation over rewrites, not a call to
`simplify`.

The numerical check is not a formality. A rewrite that sympy considers valid can still be
wrong on (0, 1): `force=True` on logcombine and powsimp lets it assume sign conditions it
cannot prove, which is usually what we want on this domain and occasionally silently
changes the function. Nothing is returned that does not reproduce the composed form on
the grid to `RTOL`, so a bad rewrite is dropped rather than believed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from .df_transforms import EPS, sympy_form
from .sr_complexity import complexity, leaf_multiplicity, outside_alphabet

RTOL = 1e-9
ATOL = 1e-12

# How many rewrites each expression is put through, reported so that "no alternative
# survived" can be told apart from "nothing was attempted".
N_REWRITES = 20

# Largest c(u) worth composing and simplifying, inclusive: expressions at or below this
# are put through the full rewrite search, larger ones are left alone.
#
# The cost is not gradual. `sp.simplify` with a custom measure explores a rewrite tree
# whose size grows with the expression, and a composed maxsize-40 front is large enough
# that a single candidate can run for minutes or fail to return at all -- the composition
# has already multiplied the tree by the transform's own nodes before the simplifier
# starts. The forms up there are also the ones least worth the wait: they are deep in the
# overfitted tail of the Pareto front, past where any of them is a claim about the DF.
# The cutoff buys back the runtime at the cost of nothing that gets reported on.
MAX_COMPLEXITY_U = 20

# PySR writes its fitted expressions in `u`; `x0` appears when variable_names is unset.
_U = sp.Symbol("u", real=True)
_PARSE_LOCALS = {
    "u": _U, "x": _U, "x0": _U,
    "log": sp.log, "exp": sp.exp, "sqrt": sp.sqrt, "asin": sp.asin, "sinh": sp.sinh,
    "square": lambda a: a**2, "cube": lambda a: a**3, "inv": lambda a: 1 / a,
    "neg": lambda a: -a, "abs": sp.Abs,
}


def default_grid(lo: float = 0.03, hi: float = 0.999, n: int = 400) -> np.ndarray:
    """The abscissa the fits were built on; pass the fold's own grid where it matters."""
    return np.linspace(float(lo), float(hi), int(n))


def parse(expr_str: str, evaluate: bool = True) -> sp.Expr:
    """Parse a PySR expression string into sympy, in the variable u.

    `evaluate=False` keeps the tree PySR actually built, so its complexity can be
    recounted as PySR counted it; sympy otherwise distributes constant factors over sums
    on parse, which is a different -- and more expensive -- expression.
    """
    return sp.sympify(str(expr_str), locals=_PARSE_LOCALS, evaluate=evaluate)


def compose(expr, transform: str) -> sp.Expr:
    """Substitute u -> T(eps_hat), giving the model as a function of eps_hat."""
    e = expr if isinstance(expr, sp.Expr) else parse(expr)
    return e.subs(_U, sympy_form(transform))


def _graft(e: sp.Expr, sym: sp.Symbol, repl: sp.Expr) -> sp.Expr:
    """Substitute without letting sympy rebuild the tree around the graft."""
    if e == sym:
        return repl
    if e.is_Atom:
        return e
    args = [_graft(a, sym, repl) for a in e.args]
    try:
        return e.func(*args, evaluate=False)
    except TypeError:
        return e.func(*args)


def _tidy(e: sp.Expr) -> sp.Expr:
    """Clear the printing artefacts of unevaluated construction, at no cost in nodes.

    Building the graft with evaluate=False preserves the tree PySR wrote, but sympy's
    printer renders an unevaluated Mul(-1, 2.55941) as `- 1*2.55941`. The stray 1 is not
    a node -- there is no literal 1 in the tree, and the counter charges nothing for it,
    since the Add branch reads a leading -1 as subtraction. Folding such a product into a
    single negative constant, and dropping any genuine unit factor, changes how the
    expression reads and not what it costs.
    """
    if e.is_Atom:
        return e
    args = [_tidy(a) for a in e.args]
    if e.is_Mul:
        kept = [a for a in args if a != sp.S.One]
        if len(kept) == 2 and kept[0] == -1 and kept[1].is_Number:
            return -kept[1]                      # Mul(-1, 2.559) -> Float(-2.559)
        if not kept:
            return sp.S.One
        if len(kept) == 1:
            return kept[0]
        args = kept
    try:
        return e.func(*args, evaluate=False)
    except TypeError:
        return e.func(*args)


def compose_raw(expr, transform: str) -> sp.Expr:
    """Compose without simplifying: g(T(eps_hat)) exactly as the substitution writes it.

    The baseline every simplification is measured against: its node count is
    c(g) + k (c(T) - 1), k = leaf_multiplicity(g, u) (u**3 counts as three leaves).
    Plain .subs would return sympy's canonical form, which can cost more.
    """
    e = expr if isinstance(expr, sp.Expr) else parse(expr, evaluate=False)
    return _graft(e, _U, sympy_form(transform))


# Stands for 1 - eps_hat, and carries the one fact sympy cannot deduce from EPS alone.
_ONE_MINUS_EPS = sp.Symbol("one_minus_eps_hat", positive=True)


def _domain_aware(fn):
    """Run a sympy simplification knowing that 1 - eps_hat is positive.

    EPS is declared positive, but the fits live on (0, 1) and nothing tells sympy the
    upper end. Without it every root and log of 1 - eps_hat has to be hedged: the
    composed `sqrt(u*u)` under sqrt_ratio comes back as sqrt(eps_hat)*Abs(sqrt(-1/(eps_hat
    - 1))), eleven nodes and an Abs the search never had. `refine` with Q.positive does
    not reach inside the radical; substituting a positive symbol for the whole subtree
    does. The symbol is put back afterwards, so the result is still a function of eps_hat.
    """
    def run(e: sp.Expr) -> sp.Expr:
        return fn(e.subs(1 - EPS, _ONE_MINUS_EPS)).subs(_ONE_MINUS_EPS, 1 - EPS)
    return run


def _measure(e: sp.Expr) -> float:
    """Complexity as a simplification objective, for sympy to minimise directly."""
    try:
        return float(complexity(e))
    except (TypeError, ValueError):
        return float("inf")


def _full_simplify(e: sp.Expr) -> sp.Expr:
    """sp.simplify, minimising PySR's node count (measure=_measure) rather than count_ops.

    Returns the input unless the result is strictly cheaper: sympy's `ratio` guard is
    skipped on some paths and can let a result come back heavier.
    """
    try:
        out = sp.simplify(e, measure=_measure, ratio=1)
    except Exception:
        return e
    return out if _measure(out) < _measure(e) else e


def _rewrites(e: sp.Expr):
    """Candidate simplifications, cheapest transformations first."""
    yield "substitution", e
    for name, fn in (
        ("full_simplify", _full_simplify),
        # sympy's own full simplification, unmodified, so that whatever `simplify` would
        # have returned is always among the candidates and the result can never be worse
        # than simply calling it. It optimises count_ops rather than the node metric, so
        # it is a candidate and not the answer.
        ("sympy_simplify", sp.simplify),
        ("cancel",        sp.cancel),
        ("together",      sp.together),
        ("radsimp",       sp.radsimp),
        ("expand",        sp.expand),
        ("powsimp",       lambda x: sp.powsimp(x, force=True)),
        ("logcombine",    lambda x: sp.logcombine(x, force=True)),
        ("expand_log",    lambda x: sp.expand_log(x, force=True)),
        ("simplify",      sp.simplify),
        ("simplify_log",  lambda x: sp.simplify(sp.expand_log(x, force=True))),
        ("simplify_pow",  lambda x: sp.simplify(sp.powsimp(x, force=True))),
        ("factor_terms",  sp.factor_terms),
        # The same rewrites again, this time knowing 1 - eps_hat > 0.
        ("dom_simplify",  _domain_aware(_full_simplify)),
        ("dom_powsimp",   _domain_aware(lambda x: sp.powsimp(x, force=True, combine="all"))),
        ("dom_powdenest", _domain_aware(lambda x: sp.powdenest(x, force=True))),
        ("dom_radsimp",   _domain_aware(sp.radsimp)),
        ("dom_logcombine", _domain_aware(lambda x: sp.logcombine(x, force=True))),
        ("dom_cancel",    _domain_aware(sp.cancel)),
    ):
        try:
            yield name, fn(e)
        except Exception:
            continue


def _agrees(candidate: sp.Expr, reference: sp.Expr, grid: np.ndarray) -> bool:
    """True if `candidate` reproduces `reference` on `grid` to within tolerance."""
    try:
        f = sp.lambdify(EPS, candidate, "numpy")
        g = sp.lambdify(EPS, reference, "numpy")
        with np.errstate(all="ignore"):
            a = np.asarray(f(grid), dtype=complex)
            b = np.asarray(g(grid), dtype=complex)
    except Exception:
        return False
    if a.shape != b.shape:
        a = np.broadcast_to(a, b.shape)
    ok = np.isfinite(b.real) & (np.abs(b.imag) < ATOL)
    if not ok.any():
        return False
    a, b = a[ok], b[ok]
    if np.any(np.abs(a.imag) > ATOL) or not np.all(np.isfinite(a.real)):
        return False
    return bool(np.allclose(a.real, b.real, rtol=RTOL, atol=ATOL))


@dataclass
class Composed:
    """One fitted expression, carried back onto eps_hat."""
    expr_u: sp.Expr             # as PySR fitted it
    expr_eps: sp.Expr           # composed and simplified
    complexity_u: int           # recounted in u, for a like-for-like delta
    complexity_eps: int         # the number to compare across transforms
    rewrite: str                # which candidate won
    outside: set                # operators the search could not have produced
    n_checked: int              # *distinct* alternatives that passed the numerical check.
                                # Zero does not mean nothing was tried: it means every
                                # rewrite returned the expression unchanged, so it is a
                                # fixed point of all of them and already simplest.
    expr_eps_in_alphabet: sp.Expr | None   # cheapest form PySR could itself have written
    complexity_eps_in_alphabet: int | None # its cost, or None if no candidate qualifies


def best_form(expr, transform: str, grid: np.ndarray | None = None) -> Composed:
    """Compose onto eps_hat and return the lowest-complexity form that is exact on the grid."""
    g = default_grid() if grid is None else np.asarray(grid, float)
    e_u = expr if isinstance(expr, sp.Expr) else parse(expr)
    # Counted on the unevaluated tree, so complexity_u reproduces PySR's own column and
    # the step up to complexity_eps is the true cost of the change of variable.
    e_u_raw = expr if isinstance(expr, sp.Expr) else parse(expr, evaluate=False)
    c_u = complexity(e_u_raw)

    # The unsimplified graft is both the numerical reference and the baseline to beat.
    reference = compose_raw(e_u_raw, transform)
    best, best_c, best_name, n_ok = reference, complexity(reference), "substitution", 0
    # Tracked separately: sympy simplifies harder than PySR ever could, and will happily
    # leave the operator set to do it -- tan_pi composes to -cot(pi*eps_hat), which is
    # cheaper but is not an expression the search could have produced. A cost quoted as a
    # PySR complexity has to be payable in PySR's own alphabet.
    in_best, in_c = (reference, best_c) if not outside_alphabet(reference) else (None, None)
    seen: set[str] = {sp.srepr(reference)}
    for name, cand in _rewrites(compose(e_u, transform)):
        key = sp.srepr(cand)
        if key in seen:
            continue
        seen.add(key)
        if not _agrees(cand, reference, g):
            continue
        n_ok += 1
        try:
            c = complexity(cand)
        except TypeError:
            continue
        if c < best_c:
            best, best_c, best_name = cand, c, name
        if not outside_alphabet(cand) and (in_c is None or c < in_c):
            in_best, in_c = cand, c

    # The graft is built unevaluated, so it prints artefacts like `- 1*2.55` even when it
    # is genuinely the cheapest form. Hand back sympy's tidy version of it where that
    # costs nothing, so the reported expression is readable and the count is unchanged.
    if best_name == "substitution":
        tidy = compose(e_u, transform)
        if complexity(tidy) == best_c and _agrees(tidy, reference, g):
            best = tidy

    # Cosmetic pass, applied whichever candidate won. It is exact by construction --
    # folding Mul(-1, c) to -c and dropping unit factors cannot change a value -- so it
    # is not put through _agrees, which would reject it wherever the composed function
    # has poles on the grid and leave `1*` in the printed form for tan_pi alone.
    stripped = _tidy(best)
    if complexity(stripped) == best_c:
        best = stripped

    return Composed(
        expr_u=e_u,
        expr_eps=best,
        complexity_u=c_u,
        complexity_eps=best_c,
        rewrite=best_name,
        outside=outside_alphabet(best),
        n_checked=n_ok,
        expr_eps_in_alphabet=in_best,
        complexity_eps_in_alphabet=in_c,
    )
