"""
Refit each functional family to the median of all 30 halos, bootstrap its parameters, and score
the Pareto front.

The CV chooses the forms; the constants quoted in the paper are refitted here on the median of
all 30 reference-scaled curves, with each occurrence of C in a family a free parameter. Forms
linear in their parameters (all every-fold families) are solved by least squares; the rest by
least_squares from several starts, with a convergence flag.

  families     only those found on every fold (--min-folds), since a global fit has no
               held-out data and would favour the most complex form. Hand-added forms
               (--extra-family) are exempt; the generalised logit A log eps + B log(1 - eps) + C
               is added by default (--no-generalised-logit), because PySR's alphabet makes
               A = -B nearly free and cannot reach it. The `source` column marks them.
  errors       halo bootstrap: resample the 30 halos, rebuild the median, refit
  front        Pareto front in (complexity, loss); score S_i = -d ln L / d C along it, with a
               constant model at complexity 1 as the anchor (marked is_anchor, not a family)
  complexity   recounted with core.sr_complexity on the refitted expression
  physicality  re-checked on the constants quoted, since a CV verdict was on a fold's constants

Reads   <cv>/pysr_families.csv, <cv>/folds.json, Halo_XX/df/scaled/df_ref<NN>_<method>_<tag>.csv
Writes  <out>/global_median.csv, family_fits.csv, pareto_front.csv, bootstrap_params.csv,
        prediction_bands*.csv, loss_vs_complexity.png   (<out> = <cv>/global_fit)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import least_squares

from ..core.physicality import EPS, check
from ..core.sr_complexity import complexity as sr_complexity
from ..io.results_paths import halo_results_root

C = sp.Symbol("C", real=True)
N_STARTS = 24          # multistart draws for a nonlinear family, on the full-data fit
FIT_MAXFEV = 20_000
IMAG_TOL = 1e-8        # |Im| past which a complex-valued form is called unusable
COND_MAX = 1e8         # Jacobian condition number past which parameters are not separable
JAC_STEP = 1e-6        # relative step for the numerical Jacobian


# ------------------------------------------------------------------------------------
# the global median
# ------------------------------------------------------------------------------------

def halo_results_for(cv: Path, override: str | None) -> Path:
    """Halo_Results root for a CV directory: two levels above <Halo_Results>/_sr/<cv name>.

    Falls back to results_paths.halo_results_root() if the CV is not under _sr/;
    --halo-results overrides both.
    """
    if override:
        return Path(override).expanduser().resolve()
    cv = cv.resolve()
    if cv.parent.name == "_sr":
        return cv.parents[1]
    return halo_results_root()


def _read_ref(root: Path, hid: int, method: str, tag: str, ref_hid: int):
    """One halo's reference-scaled curve against its own eps_hat, as build_cv_folds reads it."""
    p = root / f"Halo_{int(hid):02d}" / "df" / "scaled" / f"df_ref{ref_hid:02d}_{method}_{tag}.csv"
    if not p.exists():
        return None
    d = np.genfromtxt(p, delimiter=",", names=True)
    names = set(d.dtype.names or ())
    if "eps_hat" not in names or "log10f_ref" not in names:
        raise RuntimeError(f"{p}: expected eps_hat and log10f_ref, found {sorted(names)}")
    x, y = np.asarray(d["eps_hat"], float), np.asarray(d["log10f_ref"], float)
    m = np.isfinite(x) & np.isfinite(y)
    o = np.argsort(x[m])
    return x[m][o], y[m][o]


def global_stack(manifest: dict, root: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Every halo interpolated onto the folds.json grid, in the reference frame.

    The same grid the folds were built on, so the loss here is comparable with the CV's
    without anyone having to rescale it.
    """
    grid = np.linspace(float(manifest["grid_lo"]), float(manifest["grid_hi"]),
                       int(manifest["n_grid"]))
    method, ref_hid = str(manifest["method"]), int(manifest["ref_halo"])
    tag = str(manifest.get("tag") or manifest["shape"])
    eps_lo, eps_hi = float(manifest["eps_lo"]), float(manifest["eps_hi"])

    rows, halos = [], []
    for hid in [int(h) for h in manifest["halos"]]:
        g = _read_ref(root, hid, method, tag, ref_hid)
        if g is None:
            print(f"  halo {hid}: no df_ref{ref_hid:02d}_{method}_{tag}.csv, left out")
            continue
        x, y = g
        k = (x >= eps_lo) & (x <= eps_hi)
        if np.count_nonzero(k) < 2 or x[k][0] > grid[0] + 1e-12 or x[k][-1] < grid[-1] - 1e-12:
            print(f"  halo {hid}: does not cover the grid, left out")
            continue
        rows.append(np.interp(grid, x[k], y[k]))
        halos.append(hid)
    if len(rows) < 2:
        raise RuntimeError("fewer than two halos cover the grid; nothing to take a median of")
    return grid, np.asarray(rows, float), halos


def verify_against_cv(cv: Path, grid: np.ndarray, stack: np.ndarray,
                      halos: list[int], manifest: dict) -> None:
    """Check the curves loaded are the build this CV was cut from.

    Rebuilds fold 1's training median and compares it with the stored one, which catches a
    different grid, offsets or scaling. (A mismatched build once moved the generalised
    logit's A from 1.14 to 1.26 with no other sign.)
    """
    f1 = next((f for f in manifest["folds"] if int(f["fold"]) == 1), None)
    ref = cv / (f1 or {}).get("dir", "fold1") / "transforms" / "median_identity.csv"
    if f1 is None or not ref.exists():
        print("  (no fold1/transforms/median_identity.csv; consistency unchecked)")
        return
    m = pd.read_csv(ref)
    gx, gy = np.asarray(m["eps_hat"], float), np.asarray(m["log10f"], float)
    if gx.size != grid.size or np.max(np.abs(gx - grid)) > 1e-9:
        raise SystemExit(
            f"grid mismatch: folds.json gives {grid.size} points on "
            f"[{grid[0]:.6f}, {grid[-1]:.6f}] but {ref} is on "
            f"[{gx[0]:.6f}, {gx[-1]:.6f}]. These are different builds of the CV; point "
            f"--cv-dir at the one the families table came from.")
    idx = {h: i for i, h in enumerate(halos)}
    train = [idx[h] for h in halos if h not in set(int(t) for t in f1["test_halos"])]
    d = float(np.max(np.abs(np.median(stack[train], axis=0) - gy)))
    if d > 1e-6:
        raise SystemExit(
            f"the halo curves do not reproduce {ref.parent.parent.name}'s stored training "
            f"median (max |diff| = {d:.3e} dex). Halo_Results and this CV are different "
            f"vintages; rebuild one or point at the matching pair. --no-verify overrides.")
    print(f"  consistency: rebuilt fold-1 training median matches to {d:.1e} dex")


# ------------------------------------------------------------------------------------
# families -> parametric models
# ------------------------------------------------------------------------------------

def _letters(n: int) -> list[str]:
    """A, B, C, ... then A1, B1, ... past the alphabet, so a name is never reused."""
    out = []
    for k in range(n):
        out.append(chr(ord("A") + k % 26) + ("" if k < 26 else str(k // 26)))
    return out


def free_parameters(form: str) -> tuple[sp.Expr, list[sp.Symbol]]:
    """Give every occurrence of C its own symbol: A, B, C, ... in printed order.

    xreplace cannot, since the repeated C is one object. So C*log(eps_hat) + C*log(1 - eps_hat) + C
    becomes A + B*log(eps_hat) + C*log(1 - eps_hat), and results are stored under those letters.
    """
    e = sp.sympify(form, locals={"eps_hat": EPS, "C": C})
    counter = itertools.count()

    def walk(x):
        if x == C:
            return sp.Symbol(f"\u00a7{next(counter)}", real=True)
        if not x.args:
            return x
        return x.func(*[walk(a) for a in x.args])

    out = walk(e)
    tmp = sorted([s for s in out.free_symbols if s.name.startswith("\u00a7")],
                 key=lambda s: int(s.name[1:]))

    # Order by first appearance in the printed form, then name A, B, C, ... so the letters
    # run left to right through the expression a reader sees.
    txt = str(out)
    order = sorted(tmp, key=lambda s: (txt.find(s.name) if s.name in txt else 10**6))
    names = _letters(len(order))
    out = out.xreplace({s: sp.Symbol(n, real=True) for s, n in zip(order, names)})
    ps = [sp.Symbol(n, real=True) for n in names]

    n_c = sum(1 for _ in sp.preorder_traversal(e) if _ == C)
    if len(ps) != n_c:
        # evaluate-on-rebuild folded two parameters together (C*C, C+C); the form is
        # over-parameterised as written and the reduced one is what gets fitted.
        print(f"    note: {n_c} C's collapsed to {len(ps)} free parameters on rebuild")
    return out, ps


def is_linear(expr: sp.Expr, ps: list[sp.Symbol]) -> bool:
    """Linear in the parameters iff no derivative with respect to one still contains one."""
    if not ps:
        return True
    try:
        return all(not sp.diff(expr, p).has(*ps) for p in ps)
    except Exception:
        return False


def _callable(expr: sp.Expr, ps: list[sp.Symbol]):
    f = sp.lambdify((EPS, *ps), expr, modules=["numpy"])

    def g(x, theta):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                v = np.asarray(f(x, *theta), dtype=complex)
            except Exception:
                return np.full_like(x, np.nan)
        v = np.broadcast_to(v, x.shape)
        if not np.any(np.isfinite(v)):
            return np.full_like(x, np.nan)
        if np.nanmax(np.abs(v.imag)) > IMAG_TOL:
            return np.full_like(x, np.nan)
        return np.asarray(v.real, float)

    return g


# ------------------------------------------------------------------------------------
# fitting
# ------------------------------------------------------------------------------------

def fit_linear(expr, ps, x, y):
    """Exact least squares, using the Jacobian columns as the design matrix."""
    const = _callable(expr.xreplace({p: sp.Integer(0) for p in ps}), [])(x, ())
    cols = [_callable(sp.diff(expr, p).xreplace({q: sp.Integer(0) for q in ps}), [])(x, ())
            for p in ps]
    A = np.column_stack(cols) if cols else np.zeros((x.size, 0))
    ok = np.isfinite(const) & np.all(np.isfinite(A), axis=1) & np.isfinite(y)
    if np.count_nonzero(ok) < max(len(ps) + 1, 10):
        return None, np.inf, False
    theta, *_ = np.linalg.lstsq(A[ok], (y - const)[ok], rcond=None)
    pred = const + (A @ theta if len(ps) else 0.0)
    m = np.isfinite(pred)
    if np.count_nonzero(m) < 0.9 * x.size:
        return None, np.inf, False
    return np.asarray(theta, float), float(np.mean((pred[m] - y[m]) ** 2)), True


def _starts(n: int, rng) -> list[np.ndarray]:
    base = [np.ones(n), np.zeros(n) + 0.1, np.full(n, -1.0)]
    draws = [rng.uniform(-3, 3, n) for _ in range(max(0, N_STARTS - len(base)))]
    return base + draws


def fit_nonlinear(expr, ps, x, y, rng, starts=None):
    """least_squares from several starts; the best finite one wins.

    A start that sends the form to infinity is a normal event here, not a problem to be
    reported: the alphabet contains exp(sinh(...)) and most of the parameter space
    overflows. The warnings are silenced for the whole search so that a genuine one, from
    anywhere else in the run, is still visible.
    """
    g = _callable(expr, ps)

    def resid(theta):
        r = g(x, theta) - y
        return np.where(np.isfinite(r), r, 1e6)

    best, best_mse, ok = None, np.inf, False
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for s in (starts if starts is not None else _starts(len(ps), rng)):
            try:
                out = least_squares(resid, s, max_nfev=FIT_MAXFEV,
                                    method="lm" if len(ps) <= x.size else "trf")
            except Exception:
                continue
            pred = g(x, out.x)
            m = np.isfinite(pred)
            if np.count_nonzero(m) < 0.9 * x.size:
                continue
            mse = float(np.mean((pred[m] - y[m]) ** 2))
            if np.isfinite(mse) and mse < best_mse:
                best, best_mse, ok = np.asarray(out.x, float), mse, bool(out.success)
    return best, best_mse, ok


def fit(expr, ps, x, y, rng, starts=None):
    if is_linear(expr, ps):
        return fit_linear(expr, ps, x, y)
    return fit_nonlinear(expr, ps, x, y, rng, starts=starts)


def identifiability(expr, ps, x, theta):
    """Condition number of the Jacobian at the optimum, and its numerical rank.

    A degenerate family (e.g. C*log(C*eps + C), where two constants trade off) fits well but
    has a flat direction; its per-parameter sigmas then measure that valley, not an
    uncertainty, and the flag set from this keeps them out of tables. The prediction band
    is meaningful either way.
    """
    if not len(ps):
        return 1.0, 0          # a form with no constants: nothing to be degenerate about
    g = _callable(expr, ps)
    base = g(x, theta)
    if not np.all(np.isfinite(base)):
        return np.nan, 0
    cols = []
    for k in range(len(ps)):
        h = JAC_STEP * max(abs(theta[k]), 1.0)
        up, dn = np.array(theta, float), np.array(theta, float)
        up[k] += h
        dn[k] -= h
        cols.append((g(x, up) - g(x, dn)) / (2 * h))
    J = np.column_stack(cols)
    if not np.all(np.isfinite(J)):
        return np.nan, 0
    sv = np.linalg.svd(J, compute_uv=False)
    if sv.size == 0 or sv[0] == 0:
        return np.nan, 0
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    rank = int(np.count_nonzero(sv > sv[0] * 1e-10))
    return cond, rank


# ------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------

_CACHE: dict = {}


def _model_for(form: str):
    if form not in _CACHE:
        expr, ps = free_parameters(form)
        _CACHE[form] = (expr, ps, is_linear(expr, ps))
    return _CACHE[form]


def _boot_chunk(payload):
    """One worker's share of the replicates: resample halos, remedian, refit.

    Every nonlinear refit starts from the full-data optimum rather than from a fresh
    multistart. That is deliberate: the replicate is meant to answer "how would this fit
    have moved", and restarting the global search each time would mix that with the
    search's own scatter between basins.
    """
    form, theta, grid, stack, seed, n = payload
    expr, ps, lin = _model_for(form)
    rng = np.random.default_rng(seed)
    n_halo = stack.shape[0]
    theta = np.asarray(theta, float)
    out = []
    for _ in range(int(n)):
        idx = rng.integers(0, n_halo, n_halo)
        yb = np.median(stack[idx], axis=0)
        tb, mb, _ok = fit_linear(expr, ps, grid, yb) if lin else \
            fit_nonlinear(expr, ps, grid, yb, rng, starts=[theta])
        if tb is None or not np.isfinite(mb):
            continue
        out.append(np.concatenate([tb, [mb]]))
    return out


def bootstrap(form, theta, grid, stack, n_draws, jobs, seed):
    """n_draws halo resamples, spread over `jobs` processes."""
    if n_draws <= 0 or not len(theta):
        return np.zeros((0, len(theta))), np.zeros(0)
    jobs = max(1, int(jobs))
    per = [n_draws // jobs + (1 if k < n_draws % jobs else 0) for k in range(jobs)]
    per = [c for c in per if c]
    loads = [(form, theta, grid, stack, seed + 7919 * k, c) for k, c in enumerate(per)]
    if len(loads) == 1:
        res = [_boot_chunk(loads[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(loads)) as ex:
            res = list(ex.map(_boot_chunk, loads))
    rowsb = [r for chunk in res for r in chunk]
    if not rowsb:
        return np.zeros((0, len(theta))), np.zeros(0)
    A = np.asarray(rowsb, float)
    return A[:, :-1], A[:, -1]


# ------------------------------------------------------------------------------------
# front and score
# ------------------------------------------------------------------------------------

def pareto(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a row only if nothing cheaper has a loss as low."""
    d = df[np.isfinite(df["loss"])].sort_values(["complexity", "loss"])
    keep, best = [], np.inf
    for i, r in d.iterrows():
        if r["loss"] < best:
            keep.append(i)
            best = r["loss"]
    return d.loc[keep]


def score(front: pd.DataFrame) -> pd.DataFrame:
    """S_i = -[ln L_i - ln L_{i-1}] / [C_i - C_{i-1}], along increasing complexity."""
    f = front.sort_values("complexity").copy()
    L, Cx = f["loss"].to_numpy(float), f["complexity"].to_numpy(float)
    s = np.full(L.size, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        s[1:] = -(np.log(L[1:]) - np.log(L[:-1])) / (Cx[1:] - Cx[:-1])
    f["score"] = s
    return f


# ------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True,
                    help="the CV directory holding folds.json")
    ap.add_argument("--families", type=str, default=None,
                    help="pysr_families.csv (default: <cv-dir>/pysr_families.csv)")
    ap.add_argument("--halo-results", type=str, default=None,
                    help="the Halo_Results tree holding Halo_XX (default: two levels "
                         "above the CV dir when it sits under _sr/)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="default: <cv-dir>/global_fit")
    ap.add_argument("--bootstrap", type=int, default=100_000,
                    help="halo resamples; 0 disables the error bars")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="processes the replicates are spread over")
    ap.add_argument("--save-draws", action="store_true",
                    help="write every replicate to bootstrap_params.csv (large at 1e5)")
    ap.add_argument("--min-folds", type=str, default="all",
                    help="keep only families found on at least this many folds; 'all' "
                         "(the default) means every fold in the CV")
    ap.add_argument("--extra-family", type=str, action="append", default=None,
                    metavar="FORM",
                    help="a form to fit alongside the CV's, written in the families' own "
                         "notation with every free constant as C. Repeatable.")
    ap.add_argument("--no-generalised-logit", dest="gen_logit", action="store_false",
                    help="drop the hand-added A*log(eps) + B*log(1-eps) + C")
    ap.set_defaults(gen_logit=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-physicality", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the check that the halo curves match this CV's vintage")
    args = ap.parse_args()

    cv = Path(args.cv_dir)
    fam_path = Path(args.families) if args.families else cv / "pysr_families.csv"
    out = Path(args.out_dir) if args.out_dir else cv / "global_fit"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    manifest = json.loads((cv / "folds.json").read_text())
    root = halo_results_for(cv, args.halo_results)
    if not root.exists():
        raise SystemExit(f"no Halo_Results at {root}; pass --halo-results")
    print(f"halo curves from {root}")
    print(f"grid from {cv/'folds.json'}: {manifest['n_grid']} points on "
          f"[{manifest['grid_lo']:.4f}, {manifest['grid_hi']:.4f}]")
    grid, stack, halos = global_stack(manifest, root)
    if not args.no_verify:
        verify_against_cv(cv, grid, stack, halos, manifest)
    med = np.median(stack, axis=0)
    p16, p84 = np.percentile(stack, 16, axis=0), np.percentile(stack, 84, axis=0)
    print(f"global median over {len(halos)} halos: {halos}")
    pd.DataFrame({"eps_hat": grid, "median": med, "p16": p16, "p84": p84}).to_csv(
        out / "global_median.csv", index=False)

    fam = pd.read_csv(fam_path)
    fam["source"] = "pysr"
    n_total = int(fam["n_folds"].max()) if "n_folds" in fam.columns else 1

    # A form found on one fold is a form one training set produced, and the front it wins
    # is a front it wins by having more parameters than anyone checked out of sample. The
    # fold count is the only evidence there is about the *form* rather than its constants,
    # so it is the filter, and it is on by default.
    want = n_total if str(args.min_folds).lower() == "all" else int(args.min_folds)
    if "n_folds" in fam.columns:
        kept = fam[fam["n_folds"] >= want]
        print(f"{len(fam)} families from {fam_path}")
        print(f"  {len(kept)} found on >= {want} of {n_total} folds; "
              f"{len(fam) - len(kept)} dropped")
        fam = kept

    # Hand-added forms are not CV output and carry no fold count; they are never filtered,
    # and the `source` column keeps them distinguishable in every table downstream.
    manual = list(args.extra_family or [])
    if args.gen_logit:
        manual.append("C*log(eps_hat) + C*log(1 - eps_hat) + C")
    if manual:
        fam = pd.concat([fam, pd.DataFrame({"form": manual, "n_folds": np.nan,
                                            "source": "manual"})], ignore_index=True)
        for q in manual:
            print(f"  + hand-added: {q}")
    print()

    rows, draws, band = [], [], {}
    # The anchor: a constant, at complexity 1, so the cheapest family has a predecessor.
    rows.append(dict(family_form="C", is_anchor=True, source="anchor",
                     n_folds=np.nan, n_params=1,
                     linear=True, fit_ok=True, complexity=1, A=float(np.mean(med)),
                     loss=float(np.mean((med - np.mean(med)) ** 2)),
                     rmse=float(np.sqrt(np.mean((med - np.mean(med)) ** 2))),
                     loss_p16=np.nan, loss_p84=np.nan,
                     physicality="", n_boot=0, equation=f"{float(np.mean(med)):.8g}"))

    for _, r in fam.iterrows():
        form = str(r["form"])
        print(f"[{form[:88]}{'...' if len(form) > 88 else ''}]")
        try:
            expr, ps = free_parameters(form)
        except Exception as exc:
            print(f"    will not parse: {exc}")
            continue
        if ps:
            print(f"    as fitted: {str(expr)[:100]}")
        lin = is_linear(expr, ps)
        theta, mse, ok = fit(expr, ps, grid, med, rng)
        if theta is None:
            print("    no usable fit")
            rows.append(dict(family_form=form, form_labelled=str(expr),
                             is_anchor=False, source=r.get("source", "pysr"),
                             n_folds=r.get("n_folds", np.nan), n_params=len(ps),
                             linear=lin, fit_ok=False, complexity=np.nan,
                             loss=np.nan, rmse=np.nan, loss_p16=np.nan, loss_p84=np.nan,
                             physicality="", n_boot=0, equation=""))
            continue

        fitted = expr.xreplace({p: sp.Float(v) for p, v in zip(ps, theta)})
        cplx = int(sr_complexity(fitted))
        phys = ""
        if not args.no_physicality:
            try:
                phys = check(str(fitted)).status
            except Exception as exc:
                phys = f"error: {type(exc).__name__}"

        cond, rank = identifiability(expr, ps, grid, theta)
        degenerate = (not np.isfinite(cond)) or cond > COND_MAX or rank < len(ps)

        t0 = time.time()
        bp, bl = bootstrap(form, theta, grid, stack, int(args.bootstrap),
                           args.jobs, int(args.seed) + 104729 * len(rows))
        dt = time.time() - t0
        if len(bp):
            g = _callable(expr, ps)
            curves = np.array([g(grid, t) for t in bp])
            band[form] = (np.nanpercentile(curves, 16, axis=0),
                          np.nanpercentile(curves, 50, axis=0),
                          np.nanpercentile(curves, 84, axis=0),
                          np.nanstd(curves, axis=0, ddof=1))
            if args.save_draws:
                for t, m in zip(bp, bl):
                    draws.append(dict(form=form,
                                      **{q.name: float(v) for q, v in zip(ps, t)},
                                      loss=float(m)))
        nan = np.full(len(ps), np.nan)
        sig = bp.std(axis=0, ddof=1) if len(bp) > 1 else nan
        q16 = np.percentile(bp, 16, axis=0) if len(bp) else nan
        q50 = np.percentile(bp, 50, axis=0) if len(bp) else nan
        q84 = np.percentile(bp, 84, axis=0) if len(bp) else nan

        pull = np.abs(theta - q50) / np.where(sig > 0, sig, np.nan) if len(bp) > 1 else nan
        print(f"    c={cplx:<3} {'linear' if lin else 'nonlinear'} "
              f"{len(ps)} param  MSE={mse:.6g}  {phys}"
              + (f"  cond={cond:.2e}" if np.isfinite(cond) else "  cond=inf")
              + ("  DEGENERATE" if degenerate else "")
              + (f"  [{len(bp)} boot, {dt:.1f}s]" if len(bp) else ""))
        for k, q in enumerate(ps):
            if len(bp) > 1:
                warn = "  <-- point estimate off the band" if pull[k] > 5 else ""
                print(f"      {q.name} = {theta[k]:>13.6g}  sigma {sig[k]:<11.4g}"
                      f"  [{q16[k]:.6g}, {q84[k]:.6g}]{warn}")
            else:
                print(f"      {q.name} = {theta[k]:>13.6g}")

        # The families table spells every constant C, so a fitted p0 cannot be matched to
        # a place in the form without re-deriving the traversal order. Writing the form out
        # once with the p-names substituted removes the guesswork: p1*log(eps_hat) says
        # which constant p1 is, and it sits beside the value and the sigma.
        row = dict(family_form=form, form_labelled=str(expr), is_anchor=False,
                   source=r.get("source", "pysr"),
                   n_folds=r.get("n_folds", np.nan), n_params=len(ps),
                   linear=lin, fit_ok=bool(ok), complexity=cplx,
                   loss=float(mse), rmse=float(np.sqrt(mse)),
                   loss_p16=float(np.percentile(bl, 16)) if len(bl) else np.nan,
                   loss_p84=float(np.percentile(bl, 84)) if len(bl) else np.nan,
                   jac_cond=float(cond), jac_rank=int(rank), degenerate=bool(degenerate),
                   physicality=phys, n_boot=int(len(bp)), equation=str(fitted))
        # One column per parameter, not a packed string: sigma is the number being asked
        # for and it has to be readable without splitting a field.
        for k, q in enumerate(ps):
            row[f"{q.name}"] = float(theta[k])
            row[f"{q.name}_sigma"] = float(sig[k])
            row[f"{q.name}_p16"] = float(q16[k])
            row[f"{q.name}_p50"] = float(q50[k])
            row[f"{q.name}_p84"] = float(q84[k])
            row[f"{q.name}_pull"] = float(pull[k]) if len(bp) > 1 else np.nan
        rows.append(row)

    d = pd.DataFrame(rows)
    front = score(pareto(d))
    d["on_front"] = d.index.isin(front.index)
    d = d.merge(front[["score"]], left_index=True, right_index=True, how="left")
    d.sort_values(["complexity", "loss"]).to_csv(out / "family_fits.csv", index=False)
    front.to_csv(out / "pareto_front.csv", index=False)
    if draws:
        pd.DataFrame(draws).to_csv(out / "bootstrap_params.csv", index=False)
    if band:
        # The prediction band is identifiable even where the parameters are not, so it is
        # the uncertainty to quote for a degenerate family.
        pb = {"eps_hat": grid}
        for i, (form, (b16, b50, b84, bsd)) in enumerate(band.items()):
            pb[f"f{i}_p16"], pb[f"f{i}_p50"] = b16, b50
            pb[f"f{i}_p84"], pb[f"f{i}_sigma"] = b84, bsd
        pd.DataFrame(pb).to_csv(out / "prediction_bands.csv", index=False)
        pd.DataFrame({"index": [f"f{i}" for i in range(len(band))],
                      "form": list(band)}).to_csv(out / "prediction_bands_key.csv",
                                                  index=False)

    print(f"\nPareto front ({len(front)} of {len(d)}):")
    print(f"  {'c':>4}  {'loss':>12}  {'score':>8}  {'src':>6}  form")
    for _, r in front.iterrows():
        sc = "     --" if not np.isfinite(r["score"]) else f"{r['score']:8.4f}"
        flag = " [degenerate]" if r.get("degenerate") is True else ""
        print(f"  {int(r['complexity']):>4}  {r['loss']:12.6g}  {sc}  "
              f"{str(r.get('source', '')):>6}  {r['family_form'][:64]}{flag}")

    # The table to read. Every form here cleared the fold filter or was added by hand, so
    # the ordering is by what it costs, and the score says what the last step bought.
    rank = d[(d["fit_ok"] != False) & np.isfinite(d["loss"])].sort_values("complexity")
    scope = ("hand-added only" if want > n_total else
             f"every one found on {want} of {n_total} folds or hand-added")
    print(f"\nRanking ({len(rank)} forms, {scope}):")
    hdr = f"  {'c':>4} {'par':>4} {'loss':>11} {'rmse':>8} {'score':>8} {'fold':>5} " \
          f"{'src':>6} {'ident':>6} {'phys':>9}  form"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in rank.iterrows():
        sc = "      --" if not np.isfinite(r.get("score", np.nan)) else f"{r['score']:8.4f}"
        nf = "  -" if not np.isfinite(r.get("n_folds", np.nan)) else f"{int(r['n_folds']):3d}"
        ident = "  no" if r.get("degenerate") is True else " yes"
        print(f"  {int(r['complexity']):>4} {int(r['n_params']):>4} {r['loss']:11.5g} "
              f"{r['rmse']:8.4f} {sc} {nf:>5} {str(r.get('source','')):>6} "
              f"{ident:>6} {str(r.get('physicality','')):>9}  {r['family_form'][:60]}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        off = d[~d["on_front"] & np.isfinite(d["loss"])]
        ax.scatter(off["complexity"], off["loss"], s=22, c="0.72", label="families")
        ax.plot(front["complexity"], front["loss"], "-o", c="C0", ms=5, label="Pareto front")
        for _, r in front.iterrows():
            if np.isfinite(r["score"]):
                ax.annotate(f"{r['score']:.2f}", (r["complexity"], r["loss"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel("complexity (PySR node count)")
        ax.set_ylabel(r"MSE against the global median  [dex$^2$]")
        ax.set_title(f"Refitted on the median of {len(halos)} halos; labels are the score")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / "loss_vs_complexity.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"(no plot: {exc})")

    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
