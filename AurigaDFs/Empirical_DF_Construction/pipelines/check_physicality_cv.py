
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..core.physicality import (DECADES, LADDER_DECADES, TEST_TIMEOUT, Verdict,
                                check)

REQUIRED = ("fold", "transform", "complexity_eps")

def _row_input(r: pd.Series) -> tuple[str, str | None]:
    """The expression to sweep, and the transform it needs if it is still in u.

    `equation_eps` is what compose_pysr_fronts wrote and is what gets swept. The fallback
    exists for a table written before composition: `equation_u` together with its
    transform is the same model, and `physicality.check` will compose it.
    """
    e = r.get("equation_eps")
    if isinstance(e, str) and e.strip():
        return e, None
    return str(r["equation_u"]), str(r["transform"])


def _sweep(args: tuple[int, str, str | None, int]) -> tuple[int, dict, list]:
    """One expression, swept. Returns (row index, verdict record, ladder rows).

    Takes and returns plain data so that it can cross a process boundary: sympy
    expressions and the Side arrays are not worth pickling, and the sides are not kept.
    """
    i, equation, transform, timeout = args
    p = check(equation, transform, test_timeout=timeout)
    rec = {
        "monotone_verdict": p.monotone.value,
        "well_defined_verdict": p.well_defined.value,
        "low_energy_verdict": p.low_energy.value,
        "low_energy_limit": p.detail.get("low_energy"),
        "refuted": bool(p.refuted),
        "survives": bool(p.survives),
        "status": p.status,
        "n_abscissae": p.detail.get("n_abscissae"),
        "reach": p.detail.get("reach"),
        "monotone_detail": p.detail.get("monotone"),
        "well_defined_detail": p.detail.get("well_defined"),
        "timed_out_on": p.detail.get("timed_out_on"),
        "parse_error": p.detail.get("parse"),
    }
    return i, rec, list(p.ladder)


def _verdict_line(rec: dict) -> str:
    """The one-line reason a form was refuted, or the fact that it was not."""
    # isinstance, not truthiness: read back out of the csv this field is NaN where it was
    # None, and NaN is truthy, which would report every well-formed row as unparseable.
    if isinstance(rec.get("parse_error"), str):
        return f"does not parse: {rec['parse_error']}"
    for key, detail in (("monotone_verdict", "monotone_detail"),
                        ("well_defined_verdict", "well_defined_detail"),
                        ("low_energy_verdict", "low_energy_limit")):
        if rec[key] == Verdict.REFUTED.value:
            return str(rec[detail])
    # A timeout is not a clean sweep and must not read like one: the form was never
    # decided, and saying so here is what keeps it out of "surviving" by eye.
    if isinstance(rec.get("timed_out_on"), str):
        return f"UNDECIDED -- ran out of time on {rec['timed_out_on']}"
    return "no counterexample in the sweep"


def sweep_front(front: pd.DataFrame, jobs: int = 1, label: str = "",
                timeout: int = TEST_TIMEOUT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every row of one fold's composed table, swept.

    In the serial path progress is printed before each expression, so that a sweep which
    is taking minutes says which form it is taking them on. In the parallel path it is
    printed as each result lands, in completion order rather than row order: `map` yields
    in submission order, which means one slow row holds back every finished row behind it
    and the fold cannot report until the straggler returns. `timeout` bounds how long a
    straggler can be.
    """
    missing = [c for c in REQUIRED if c not in front.columns]
    if not ({"equation_eps", "equation_u"} & set(front.columns)):
        missing.append("equation_eps or equation_u")
    if missing:
        raise SystemExit(f"composed table is missing {missing}")
    if not len(front):
        return pd.DataFrame(), pd.DataFrame()

    work = [(i, *_row_input(front.iloc[i]), timeout) for i in range(len(front))]
    recs: dict[int, dict] = {}
    ladders: dict[int, list] = {}

    def note(i: int) -> str:
        r = front.iloc[i]
        return (f"  {label}[{i + 1}/{len(work)}] {r['transform']} "
                f"c(eps)={int(r['complexity_eps'])}: {str(work[i][1])[:64]}")

    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(_sweep, w) for w in work]
            for done, fut in enumerate(as_completed(futures), start=1):
                i, rec, lad = fut.result()
                recs[i], ladders[i] = rec, lad
                print(f"{note(i)} [{done}/{len(work)} done]"
                      f"\n      -> {_verdict_line(rec)}", flush=True)
    else:
        for w in work:
            print(note(w[0]), flush=True)
            i, rec, lad = _sweep(w)
            recs[i], ladders[i] = rec, lad
            print(f"      -> {_verdict_line(rec)}", flush=True)

    keep = [c for c in ("fold", "transform", "complexity_u", "complexity_eps",
                        "complexity_eps_in_alphabet", "loss", "equation_eps",
                        "equation_u") if c in front.columns]
    out = pd.concat(
        [front[keep].reset_index(drop=True),
         pd.DataFrame([recs[i] for i in range(len(work))])], axis=1)
    out = out.rename(columns={"loss": "train_loss"})

    lad_rows = []
    for i in range(len(work)):
        r = front.iloc[i]
        for eps_label, value in ladders[i]:
            lad_rows.append({"fold": r["fold"], "transform": r["transform"],
                             "complexity_eps": int(r["complexity_eps"]),
                             "eps_hat": eps_label, "log10_f": value})
    return out, pd.DataFrame(lad_rows)


def _report(fold: int, out: pd.DataFrame) -> None:
    """What the sweep found, in the terms the verdicts are actually defined in."""
    if not len(out):
        print(f"\nfold {fold}: no expressions in the composed table")
        return
    n = len(out)
    refuted = out[out["refuted"]]
    surv = out[out["survives"]]
    mono = int((out["monotone_verdict"] == Verdict.REFUTED.value).sum())
    wd = int((out["well_defined_verdict"] == Verdict.REFUTED.value).sum())
    le = int((out["low_energy_verdict"] == Verdict.REFUTED.value).sum())
    proved = int((out["low_energy_verdict"] == Verdict.HOLDS.value).sum())
    undec = int((out["low_energy_verdict"] == Verdict.UNDECIDED.value).sum())
    bad_parse = int(out["parse_error"].map(lambda v: isinstance(v, str)).sum())
    timed = out[out["timed_out_on"].map(lambda v: isinstance(v, str))]
    print(f"\nfold {fold}: {len(surv)} of {n} expressions survive the sweep")
    print(f"  refuted: {mono} on monotonicity, {wd} on well-definedness, "
          f"{le} on the low-energy limit"
          + (f", {bad_parse} did not parse" if bad_parse else ""))
    print(f"  low-energy limit: {proved} proved to vanish, {le} proved not to, "
          f"{undec} undecided by sympy")
    # Surviving and undecided are different things and the count has to say so: a form
    # that ran out of time is not refuted, so it lands in the surviving count above
    # without a single condition having been checked on it.
    if len(timed):
        print(f"  {len(timed)} ran out of time; {int((timed['status'] == 'undecided').sum())} "
              f"of those are undecided and do NOT survive:")
        for _, r in timed.iterrows():
            print(f"    c(eps)={int(r['complexity_eps']):>3} {r['transform']:<16} "
                  f"timed out on {r['timed_out_on']}")
    if len(refuted):
        print("  what was refuted, and where:")
        for _, r in refuted.iterrows():
            print(f"    c(eps)={int(r['complexity_eps']):>3} {r['transform']:<16} "
                  f"{_verdict_line(r.to_dict())}")
    if len(surv):
        cheap = surv.loc[surv["complexity_eps"].idxmin()]
        print(f"  cheapest surviving form: c(eps)={int(cheap['complexity_eps'])} "
              f"({cheap['transform']}) {cheap.get('equation_eps')}")
    # max/dropna rather than iloc[0]: a row that failed to parse was never swept and
    # carries no abscissa count or reach, and must not be what the reach is quoted from.
    reach = out["reach"].dropna()
    n_abs = out["n_abscissae"].max()
    print(f"  surviving is not holding: {int(n_abs) if n_abs == n_abs else 0} abscissae "
          f"over {reach.iloc[0] if len(reach) else 'no swept row'} found no "
          f"counterexample, which is not a proof that none exists.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True)
    ap.add_argument("--composed", type=str, default="pysr_composed.csv",
                    help="per-fold composed table to sweep (default pysr_composed.csv)")
    ap.add_argument("--out", type=str, default="pysr_physicality.csv")
    ap.add_argument("--ladder", type=str, default="pysr_physicality_ladder.csv")
    ap.add_argument("--jobs", type=int, default=1,
                    help="expressions swept in parallel; verdicts do not depend on it")
    ap.add_argument("--timeout", type=int, default=TEST_TIMEOUT,
                    help=f"seconds each numeric phase of one expression gets before it "
                         f"reports undecided (default {TEST_TIMEOUT})")
    args = ap.parse_args()

    cv = Path(args.cv_dir).expanduser().resolve()
    n_folds = 0
    for fold_dir in sorted(cv.glob("fold*")):
        composed = fold_dir / args.composed
        if not composed.exists():
            continue
        n_folds += 1
        front = pd.read_csv(composed)
        j = int(front["fold"].iloc[0])
        print(f"\nsweeping fold {j}: {len(front)} expressions over "
              f"1e-{DECADES} to 1 - 1e-{DECADES}", flush=True)
        out, ladder = sweep_front(front, jobs=args.jobs, label=f"fold {j} ",
                                  timeout=args.timeout)
        out.to_csv(fold_dir / args.out, index=False)
        ladder.to_csv(fold_dir / args.ladder, index=False)
        _report(j, out)
        print(f"  -> {fold_dir / args.out}")
        print(f"  -> {fold_dir / args.ladder} "
              f"(log10 f down {LADDER_DECADES} decades; evidence, not a verdict)")

    if not n_folds:
        raise FileNotFoundError(
            f"no {args.composed} under {cv}/fold*/ -- run compose_pysr_fronts first")
    print(f"\nswept {n_folds} folds. This filter spends no held-out evidence -- the "
          f"verdict is a\nproperty of the expression alone -- so it is the one that may "
          f"be applied before\nevaluate_pysr_cv opens the test curves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
