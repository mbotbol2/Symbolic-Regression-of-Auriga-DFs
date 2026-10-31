"""
Carry every fitted expression in a CV directory back onto eps_hat and recount its complexity.

Reads   <cv>/fold<j>/pysr/<transform>/equations.csv
Writes  <cv>/fold<j>/pysr_composed.csv
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sympy as sp

from ..core.df_transforms import EPS
from ..core.sr_complexity import complexity
from ..core.sr_simplify import (
    MAX_COMPLEXITY_U, N_REWRITES, best_form, default_grid, parse)


def fold_grid(cv: Path) -> np.ndarray:
    """The abscissa the folds were built on, so the numerical check uses the real domain."""
    manifest = cv / "folds.json"
    if not manifest.exists():
        return default_grid()
    m = json.loads(manifest.read_text())
    lo = float(m.get("grid_lo", m.get("eps_lo", 0.03)))
    hi = float(m.get("grid_hi", m.get("eps_hi", 0.999)))
    return np.linspace(lo, hi, int(m.get("n_grid", 400)))


def _reparsed_cost(expr) -> int:
    """What the printed form costs when read back in, which is not always what it costs.

    sympy distributes a constant over a sum on parse, so the 9-node `(A + B)*c` PySR built
    is read back as the 11-node `A*c + B*c`. The reported complexity is the cost of the
    tree that was actually chosen -- the factored form is a real expression in the search
    alphabet, and it is the one PySR charged 9 for -- but a reader who re-parses the
    string gets a larger number, so the two are written side by side rather than leaving
    the discrepancy to be discovered.
    """
    try:
        return complexity(sp.sympify(str(expr), locals={"eps_hat": EPS}))
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True)
    ap.add_argument("--max-complexity", type=int, default=MAX_COMPLEXITY_U,
                    help=f"largest c(u) to compose, inclusive (default {MAX_COMPLEXITY_U}); "
                         f"simplification of the largest composed forms is slow, sometimes "
                         f"unboundedly so, and rarely informative. 0 or less lifts the cut.")
    ap.add_argument("--out", type=str, default="pysr_composed.csv",
                    help="filename written inside each fold<j>/ (default "
                         "pysr_composed.csv). The composed front is a per-fold artefact "
                         "and belongs beside the equations.csv it came from.")
    ap.add_argument("--pooled", type=str, default=None, metavar="NAME",
                    help="also write every fold concatenated to <cv>/NAME. Off by "
                         "default: one table holding all folds is a table with every "
                         "halo's test set in view, which is a report on selection already "
                         "made rather than an input to it.")
    args = ap.parse_args()

    cv = Path(args.cv_dir).expanduser().resolve()
    found = sorted(cv.glob("fold*/pysr/*/equations.csv"))
    if not found:
        raise FileNotFoundError(
            f"no equations.csv under {cv}/fold*/pysr/*/ -- run the fits first")

    grid = fold_grid(cv)
    # <= 0 means no cut, so a run over a whole front is still one flag away.
    cut = int(args.max_complexity) if int(args.max_complexity) > 0 else None
    rows, mismatches, skipped = [], [], []
    # fold number -> the directory it was read from, so each composed table is written
    # back beside its own equations.csv whatever that directory is called.
    fold_dirs: dict[int, Path] = {}
    for path in found:
        transform = path.parent.name
        fold = int(re.sub(r"\D", "", path.parents[2].name))
        fold_dirs[fold] = path.parents[2]
        eq = pd.read_csv(path)
        # Announced before the work, not after: simplifying one maxsize-40 front takes
        # minutes, and a run whose first output arrives that late is indistinguishable
        # from one that has hung.
        n_todo = len(eq) if cut is None else int((eq["complexity"].astype(int) <= cut).sum())
        print(f"  {path.parents[2].name}/{transform}: {len(eq)} expressions"
              + ("" if n_todo == len(eq) else f", {n_todo} at or below c(u)={cut}"),
              flush=True)
        i = 0
        for _, r in eq.iterrows():
            c_pysr = int(r["complexity"])
            if cut is not None and c_pysr > cut:
                skipped.append((fold, transform, c_pysr))
                continue
            i += 1
            started = time.time()
            # PySR's own count describes the tree PySR wrote, so parse that column.
            c_u = complexity(parse(r["equation"], evaluate=False))
            if c_u != c_pysr:
                mismatches.append((path, r["equation"], c_pysr, c_u))
            b = best_form(r["equation"], transform, grid)
            # The same pipeline with no transform: identity composition is a rename, so
            # this is the fitted expression simplified exactly as the composed one was.
            c_u_simplified = best_form(r["equation"], "identity", grid).complexity_eps
            rows.append({
                "fold": fold,
                "transform": transform,
                "complexity_u": c_pysr,
                "complexity_u_simplified": c_u_simplified,
                "complexity_eps": b.complexity_eps,
                "cost_of_transform": b.complexity_eps - c_u_simplified,
                "candidates_verified": b.n_checked,
                "loss": float(r["loss"]) if "loss" in eq.columns else np.nan,
                "equation_u": str(r["equation"]),
                "equation_eps": str(b.expr_eps),
                "rewrite": b.rewrite,
                "outside_alphabet": " ".join(sorted(b.outside)),
                "complexity_eps_reparsed": _reparsed_cost(b.expr_eps),
                "complexity_eps_in_alphabet": b.complexity_eps_in_alphabet,
                "equation_eps_in_alphabet": (
                    "" if b.expr_eps_in_alphabet is None else str(b.expr_eps_in_alphabet)),
            })
            print(f"    [{i}/{n_todo}] c(u)={c_pysr:<3} -> c(eps)={b.complexity_eps:<4}"
                  f" {time.time() - started:5.1f}s  via {b.rewrite}", flush=True)

    if mismatches:
        for path, expr, c_pysr, c_u in mismatches[:10]:
            print(f"complexity mismatch in {path}: {expr!r} PySR={c_pysr} recounted={c_u}")
        raise SystemExit(
            f"{len(mismatches)} expressions do not reproduce PySR's complexity; the parse "
            f"does not describe what was fitted, so the composed costs cannot be trusted")

    # One table per fold, written beside the front it was composed from. Nothing here is
    # aggregated across folds: see the module docstring for why the concatenation is the
    # thing being avoided rather than the layout being tidied.
    frame = pd.DataFrame(rows)
    for j in sorted(frame["fold"].unique()):
        # mergesort, because the sort key is not a total order -- two expressions can tie
        # on (transform, complexity_eps). Quicksort would order those two differently from
        # run to run and diff the CSV for no reason.
        g = (frame[frame["fold"] == j]
             .sort_values(["transform", "complexity_eps", "complexity_u"],
                          kind="mergesort")
             .reset_index(drop=True))
        dest = fold_dirs[int(j)] / args.out
        g.to_csv(dest, index=False)
        print(f"\nfold {int(j)}: {len(g)} expressions -> {dest}")
        print(f"  {'transform':<16}{'n':>4}{'median c(u)':>13}{'median c(eps)':>15}"
              f"{'median extra':>14}")
        for t, gt in g.groupby("transform"):
            print(f"  {t:<16}{len(gt):>4}{gt['complexity_u_simplified'].median():>13.0f}"
                  f"{gt['complexity_eps'].median():>15.0f}"
                  f"{gt['cost_of_transform'].median():>14.0f}")

    print(f"\n{len(frame)} expressions from {len(found)} runs -> "
          f"{frame['fold'].nunique()} per-fold tables under {cv}")
    if args.pooled:
        dest = cv / args.pooled
        (frame.sort_values(["transform", "fold", "complexity_eps"], kind="mergesort")
              .reset_index(drop=True).to_csv(dest, index=False))
        print(f"concatenated table also written -> {dest}\n"
              f"  every fold in one frame. A form selected by reading down it has been "
              f"selected on all 30\n  halos, and no held-out curve is left to test it "
              f"against.")
    if skipped:
        # Said plainly, because the tables are now a subset of the fronts rather than all
        # of them, and a reader counting rows against equations.csv has to know why.
        print(f"{len(skipped)} expressions above c(u)={cut} were not composed "
              f"(c(u) up to {max(c for _, _, c in skipped)}); "
              f"pass --max-complexity 0 to compose them too")

    # What follows is evidence about the simplifier, not about the DF: how often sympy's
    # parse disagrees with the tree that was chosen, how often it had to leave the search
    # alphabet, how hard it worked. None of it carries a loss, so none of it can steer a
    # choice of functional form, and it is reported over every fold at once.
    reparse = frame[frame["complexity_eps_reparsed"] != frame["complexity_eps"]]
    if len(reparse):
        print(f"\n{len(reparse)} expressions cost more when the printed form is re-parsed "
              f"(sympy distributes a constant over a sum); reported cost is the tree that "
              f"was chosen:")
        for _, r in reparse.head(5).iterrows():
            print(f"  {int(r['complexity_eps'])} vs {int(r['complexity_eps_reparsed'])} "
                  f"re-parsed:  {r['equation_eps']}")
    unpayable = frame[frame["complexity_eps_in_alphabet"].isna()]
    if len(unpayable):
        for t, g in unpayable.groupby("transform"):
            print(f"\n{len(g)} {t} expressions have no form inside PySR's operator set; "
                  f"their cost is not comparable with the others")

    simplified = frame[frame["complexity_u_simplified"] != frame["complexity_u"]]
    print(f"\nsimplification changed the fitted expression in {len(simplified)} of "
          f"{len(frame)} cases" + (":" if len(simplified) else
          " -- PySR's fronts are already minimal under this pipeline"))
    for _, r in simplified.head(8).iterrows():
        print(f"  fold{int(r['fold'])}/{r['transform']}: "
              f"{int(r['complexity_u'])} -> {int(r['complexity_u_simplified'])}  "
              f"{r['equation_u']}")

    # Evidence about the simplification, rather than a claim about it. Nothing winning is
    # not the same as nothing being tried: an expression already in its simplest form has
    # every candidate rejected for being no cheaper, and `candidates_verified` says how
    # many distinct rewrites were generated and checked against the grid to establish it.
    print(f"\neach expression was put through {N_REWRITES} rewrites, including a plain "
          f"sympy.simplify;\n{frame['candidates_verified'].min()}-"
          f"{frame['candidates_verified'].max()} distinct alternatives survived "
          f"verification on the grid (median "
          f"{frame['candidates_verified'].median():.0f}).\nZero means every rewrite "
          f"returned the expression unchanged -- a fixed point, already simplest.")
    by_rewrite = frame["rewrite"].value_counts()
    print("\nwhich rewrite produced the reported form:")
    for name, n in by_rewrite.items():
        print(f"  {name:<16}{n:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
