"""
Collapse each fold's front to one point per functional family, then report what folds agree on.

The alphabet writes one function more than one way (logit at c=10 and sqrt_ratio at c=12 are
the same model), so rows are grouped by family: every fitted Float becomes a free parameter
and two rows are one family if their canonical forms are symbolically equal. Integers stay,
so the 1 in (1 - eps_hat) is shape, not a parameter. Canonicalisation:

  log arguments   oriented so the denominator is positive at eps_hat = 1/2
  expand_log      pulls sqrt out of a log as a rational factor
  content         a rational factor beside a parameter is absorbed into it
  collapse        every eps-free group with a parameter becomes one parameter

Equality is decided by a capped sympy.simplify; a pair it cannot decide is reported as
undecided, not merged. --numeric-audit lists pairs equal on a grid that were not merged.

Each fold's front ranks families by held-out mean_mse (from evaluate_pysr_cv) by default,
which spends the test halos on selection; --scores none --loss-col train_loss gives the
training-side front. Fronts are built per fold and only then compared.

Reads   <cv>/fold<j>/pysr_physicality.csv, <cv>/fold<j>/pysr_holdout_summary.csv
Writes  <cv>/fold<j>/pysr_front.csv   that fold's family-deduplicated front
        <cv>/pysr_families.csv        the families, and which folds found them
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import sympy as sp

from ..core.physicality import EPS, _capped, _LimitTimeout, as_expr

C = sp.Symbol("C", real=True)
HALF = sp.Rational(1, 2)
CANON_TIMEOUT = 60     # seconds one expression gets to canonicalise
SAME_TIMEOUT = 20      # seconds one pair gets to be decided
AUDIT_RTOL = 1e-4      # the value-based check --numeric-audit reports against
AUDIT_MAG = 1e30       # magnitude past which a row is left out of the audit
AUDIT_TIMEOUT = 10     # seconds one expression gets to be evaluated for it


def _is_param(x) -> bool:
    return any(s.name.startswith("c") for s in x.free_symbols)


def _sign_at_half(x) -> int:
    """Sign of a subexpression in the middle of the interval, 0 if undecidable."""
    try:
        v = complex(sp.N(x.subs(EPS, HALF), 20))
    except Exception:
        return 0
    if abs(v.imag) > 1e-12 or v.real == 0:
        return 0
    return 1 if v.real > 0 else -1


def normalise_logs(e):
    """One log argument, one spelling: denominator positive on the interval."""
    def fix(x):
        a = sp.cancel(sp.together(x.args[0]))
        n, d = sp.fraction(a)
        if _sign_at_half(d) < 0:
            n, d = -n, -d
        return sp.log(n / d)
    return e.replace(lambda x: x.func is sp.log, fix)


def parameterise(e):
    """Every fitted constant becomes a free symbol; integers and rationals are form."""
    return e.xreplace({a: sp.Symbol(f"c{n}", real=True) for n, a in
                       enumerate(sorted(e.atoms(sp.Float), key=sp.default_sort_key))})


def collapse(e):
    """Fold each eps-free group carrying a parameter into one parameter, to a fixed point."""
    n = [0]

    def fresh():
        n[0] += 1
        return sp.Symbol(f"k{n[0]}", real=True)

    def walk(x):
        if not x.args:
            return x
        if EPS not in x.free_symbols:
            return fresh() if _is_param(x) else x
        if x.func in (sp.Add, sp.Mul):
            free = [a for a in x.args if EPS not in a.free_symbols]
            rest = [walk(a) for a in x.args if EPS in a.free_symbols]
            if free and any(_is_param(a) for a in free):
                if x.func is sp.Mul:
                    _, body = sp.Mul(*rest).as_content_primitive()
                    return sp.Mul(fresh(), body)
                return sp.Add(fresh(), *rest)
            return x.func(*free, *rest)
        return x.func(*[walk(a) for a in x.args])

    prev = None
    while prev != e:
        prev, e = e, walk(e)
    return e


def canonical(equation: str, timeout: int = CANON_TIMEOUT):
    """The family this expression belongs to, as an expression in one free constant."""
    try:
        with _capped(timeout):
            e = normalise_logs(as_expr(str(equation), None))
            e = parameterise(e)
            e = sp.powsimp(sp.expand_log(e, force=True), force=True)
            e = normalise_logs(e)
            e = collapse(e)
            e = e.xreplace({v: C for v in e.free_symbols if v != EPS})
            _, e = e.as_content_primitive()
            return e
    except (_LimitTimeout, Exception):
        return None


def shape(e) -> tuple:
    """A cheap invariant: two forms with different ones cannot be the same family."""
    if e is None:
        return ("none",)
    return (tuple(sorted(str(f.func) for f in e.atoms(sp.Function))),
            e.count(C), sp.count_ops(e))


def same(a, b, timeout: int = SAME_TIMEOUT) -> bool | None:
    """True, False, or None where sympy could not decide inside the cap."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        with _capped(timeout):
            return sp.simplify(a - b) == 0
    except _LimitTimeout:
        return None
    except Exception:
        return False


def families(df: pd.DataFrame, forms: dict) -> tuple[dict, list, list]:
    """Union rows whose canonical forms agree. Returns class map, merges, undecided pairs."""
    parent = {i: i for i in df.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merges, undecided = [], []
    # Compare only within a shape bucket: the invariant is free and rules out most pairs
    # before simplify is asked anything.
    buckets: dict = {}
    for i in df.index:
        buckets.setdefault(shape(forms[i]), []).append(i)
    for members in buckets.values():
        for i, j in itertools.combinations(members, 2):
            if find(i) == find(j):
                continue
            verdict = same(forms[i], forms[j])
            if verdict is True:
                parent[find(i)] = find(j)
                merges.append((i, j))
            elif verdict is None:
                undecided.append((i, j))
    return {i: find(i) for i in df.index}, merges, undecided


def pareto(df: pd.DataFrame, loss_col: str) -> pd.DataFrame:
    """The non-dominated rows: nothing is both simpler and better."""
    d = df.sort_values(["complexity_eps", loss_col]).reset_index(drop=True)
    best, keep = None, []
    for k, r in d.iterrows():
        if best is None or r[loss_col] < best:
            keep.append(k)
            best = r[loss_col]
    return d.loc[keep].reset_index(drop=True)


def numeric_audit(df: pd.DataFrame, forms: dict, cls: dict) -> list:
    """Pairs that agree as curves but that the symbolic test did not merge.

    A completeness net for sympy, not a decision: it merges nothing, it only names where a
    rewrite may be missing. Both guards below exist because this front contains rows that
    reach 1e+118563353500526... near eps_hat = 0 -- the fold-2 log_log_ratio pair. mpmath
    represents those happily and then spends unbounded time doing arithmetic on numbers
    whose exponents have hundreds of digits, which is what stalled this function for 49
    minutes when it had neither guard. A row that large is dominated by its overflow
    behaviour anyway and there is nothing useful to compare it against, so it is skipped.
    """
    import mpmath as mpm
    mpm.mp.dps = 30
    xs = [mpm.mpf(10) ** (-6 + 6 * k / 20) for k in range(21)]
    xs += [1 - x for x in xs if 0 < 1 - x < 1]
    vals: dict = {}
    for i in df.index:
        try:
            with _capped(AUDIT_TIMEOUT):
                f = sp.lambdify(EPS, as_expr(str(df.at[i, "equation_eps"]), None), "mpmath")
                v = []
                for x in xs:
                    z = f(x)
                    if not mpm.isfinite(z) or abs(z) > AUDIT_MAG:
                        raise ValueError("out of audit range")
                    v.append(z)
            vals[i] = v
        except (_LimitTimeout, Exception):
            vals[i] = None
    out = []
    for i, j in itertools.combinations(df.index, 2):
        if cls[i] == cls[j] or vals[i] is None or vals[j] is None:
            continue
        worst = max(abs(a - b) / max(abs(a), abs(b), mpm.mpf(1))
                    for a, b in zip(vals[i], vals[j]))
        if float(worst) <= AUDIT_RTOL:
            out.append((i, j, float(worst)))
    return out


def _label(r) -> str:
    return f"{r['transform']} c={int(r['complexity_eps'])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True)
    ap.add_argument("--physicality", type=str, default="pysr_physicality.csv")
    ap.add_argument("--loss-col", type=str, default="mean_mse",
                    help="column the front trades against: a held-out column from --scores "
                         "(mean_mse, worst_mse) or train_loss")
    ap.add_argument("--scores", type=str, default="pysr_holdout_summary.csv", metavar="FILE",
                    help="per-fold holdout summary from evaluate_pysr_cv to join in; "
                         "'none' skips the join and leaves only train_loss")
    ap.add_argument("--survivors-only", action="store_true", default=True)
    ap.add_argument("--all-rows", dest="survivors_only", action="store_false")
    ap.add_argument("--numeric-audit", action="store_true",
                    help="also report pairs that agree as curves but were not merged")
    ap.add_argument("--front-out", type=str, default="pysr_front.csv")
    ap.add_argument("--families-out", type=str, default="pysr_families.csv")
    args = ap.parse_args()

    cv = Path(args.cv_dir).expanduser().resolve()
    fronts = []
    for fold_dir in sorted(cv.glob("fold*")):
        src = fold_dir / args.physicality
        if not src.exists():
            continue
        d = pd.read_csv(src)
        j = int(d["fold"].iloc[0])
        if args.survivors_only:
            d = d[d["survives"]].copy()
        d = d[d["equation_eps"].notna()].reset_index(drop=True)

        # The held-out columns live in a different table, keyed the way evaluate_pysr_cv
        # aggregated them. Joining rather than recomputing keeps this file out of the
        # scoring business entirely: it trades against whatever column it is handed.
        if args.scores and args.scores.lower() != "none":
            sp_ = fold_dir / args.scores
            if not sp_.exists():
                raise FileNotFoundError(f"{sp_} not found -- run evaluate_pysr_cv first")
            sc = pd.read_csv(sp_)
            keys = [k for k in ("transform", "complexity_u", "complexity_eps")
                    if k in d.columns and k in sc.columns]
            before = len(d)
            d = d.merge(sc, on=keys, how="left", suffixes=("", "_score"))
            assert len(d) == before, "score join changed the row count -- keys are not unique"

        if args.loss_col not in d.columns:
            raise SystemExit(f"--loss-col {args.loss_col!r} is not a column; available: "
                             f"{sorted(c for c in d.columns if 'loss' in c or 'mse' in c or 'skill' in c)}")
        # A row with no loss cannot sit on a front. Dropping is reported, never silent.
        n_na = int(d[args.loss_col].isna().sum())
        if n_na:
            print(f"  {n_na} row(s) have no {args.loss_col} and are left off the front")
            d = d[d[args.loss_col].notna()].reset_index(drop=True)
        if "min_valid" in d.columns and "n_points" in d.columns:
            part = d[d["min_valid"] < d["n_points"]]
            if len(part):
                print(f"  WARNING {len(part)} row(s) scored on a partial grid "
                      f"(min_valid < n_points) -- their loss is not comparable")

        print(f"\nfold {j}: {len(d)} rows to reduce", flush=True)
        if not len(d):
            continue

        forms = {i: canonical(d.at[i, "equation_eps"]) for i in d.index}
        n_bad = sum(v is None for v in forms.values())
        cls, merges, undecided = families(d, forms)
        d["family"] = [sp.srepr(forms[i]) if forms[i] is not None else f"UNCANONICAL:{i}"
                       for i in d.index]
        d["family_form"] = [str(forms[i]) if forms[i] is not None else None
                            for i in d.index]
        d["eq_class"] = [cls[i] for i in d.index]
        print(f"  {len(d)} rows -> {len(set(cls.values()))} families"
              + (f"; {n_bad} would not canonicalise" if n_bad else ""))
        for i, k in merges:
            print(f"    merged: {_label(d.loc[i])} == {_label(d.loc[k])}"
                  f"   {d.at[i, 'family_form']}")
        for i, k in undecided:
            print(f"    UNDECIDED: {_label(d.loc[i])} vs {_label(d.loc[k])} "
                  f"-- sympy did not settle it in {SAME_TIMEOUT}s")
        if args.numeric_audit:
            for i, k, dev in numeric_audit(d, forms, cls):
                print(f"    audit: {_label(d.loc[i])} and {_label(d.loc[k])} agree as "
                      f"curves to {dev:.1e} but were NOT merged symbolically")

        rep = (d.sort_values(["complexity_eps", args.loss_col])
                 .groupby("eq_class", as_index=False).first())
        print(f"  Pareto: {len(pareto(d, args.loss_col))} points before dedup, "
              f"{len(pareto(rep, args.loss_col))} after")
        front = pareto(rep, args.loss_col)
        front.to_csv(fold_dir / args.front_out, index=False)
        print(f"  -> {fold_dir / args.front_out}")
        fronts.append(front)

    if not fronts:
        raise FileNotFoundError(f"no {args.physicality} under {cv}/fold*/")

    # The merge, and only now: every front above was built without seeing another fold.
    allf = pd.concat(fronts, ignore_index=True)
    fam = (allf.groupby("family")
                .agg(form=("family_form", "first"),
                     folds=("fold", lambda s: ",".join(str(x) for x in sorted(set(s)))),
                     n_folds=("fold", "nunique"),
                     c_min=("complexity_eps", "min"), c_max=("complexity_eps", "max"),
                     loss_min=(args.loss_col, "min"), loss_max=(args.loss_col, "max"))
                .reset_index()
                .sort_values(["n_folds", "loss_min"], ascending=[False, True]))
    n_total = allf["fold"].nunique()
    print(f"\n{len(fam)} families across {n_total} folds; "
          f"{int((fam['n_folds'] == n_total).sum())} appear on every fold's front")
    for _, r in fam.iterrows():
        print(f"  [{r['n_folds']}/{n_total}] c={int(r['c_min'])}-{int(r['c_max'])} "
              f"{args.loss_col} {r['loss_min']:.5f}-{r['loss_max']:.5f}  {r['form']}")

    # Written before the re-check below: that step can be slow, and nothing it finds
    # changes this table -- it only reports.
    fam.to_csv(cv / args.families_out, index=False)
    print(f"\n-> {cv / args.families_out}")

    # Check again, across folds: a family key is only as good as the canonicalisation that
    # produced it, and a rewrite that failed inside one fold would show up here as two
    # families that are really one. Parsed back with the same symbols canonical() used,
    # and compared only within a shape bucket, as families() does -- without that every
    # pair goes to simplify, which on 32 families is 496 calls of up to SAME_TIMEOUT each.
    print("\nre-checking the merged families against each other:", flush=True)
    reps = list(fam["form"])
    exprs = [sp.sympify(f, locals={"eps_hat": EPS, "C": C}) if isinstance(f, str) else None
             for f in reps]
    buckets: dict = {}
    for a, e in enumerate(exprs):
        buckets.setdefault(shape(e), []).append(a)
    pairs = [p for m in buckets.values() for p in itertools.combinations(m, 2)]
    print(f"  {len(pairs)} pair(s) share a shape and are compared", flush=True)
    found = False
    for a, b in pairs:
        verdict = same(exprs[a], exprs[b])
        if verdict is True:
            found = True
            print(f"  {reps[a]}\n  == {reps[b]}")
        elif verdict is None:
            print(f"  UNDECIDED: {reps[a]}\n         vs {reps[b]}")
    if not found:
        print("  none -- every family on the merged front is a distinct form")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
