"""
Score every composed expression against the halos its fold held out.

Each held-out curve is raw, in its halo's units; it is moved into the reference frame with
log10 f_ref = log10 f_raw + to_ref_offset_dex[halo] (folds.json) and compared with the
expression. The baseline is the training median (mse_vs_train_median in cv_summary.csv), and
`skill` = 1 - mse/baseline is positive where the expression beats it.

Rows are ordered by complexity, not score: these numbers report the error of forms chosen
upstream, and choosing by them would fit to the test halos. An expression undefined on part of
the grid is kept, with n_valid < n_points and mse over the points that survive.

Reads   <cv>/folds.json, <cv>/cv_summary.csv (rebuilt if absent),
        <cv>/fold<j>/pysr_composed.csv, <cv>/fold<j>/test/test_halo<hid>.csv
Writes  <cv>/fold<j>/pysr_holdout_scores.csv    one row per (expression, held-out halo)
        <cv>/fold<j>/pysr_holdout_summary.csv   one row per expression
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from ..core.df_transforms import EPS


def _fold_grid(cv: Path) -> np.ndarray:
    """The abscissa every fold was built on; the test curves are already written on it."""
    m = json.loads((cv / "folds.json").read_text())
    return np.linspace(float(m["grid_lo"]), float(m["grid_hi"]), int(m["n_grid"])), m


def _evaluate(expr_str: str, grid: np.ndarray) -> np.ndarray:
    """The expression on the grid, with non-finite and complex points marked NaN."""
    try:
        f = sp.lambdify(EPS, sp.sympify(expr_str, locals={"eps_hat": EPS}), "numpy")
        with np.errstate(all="ignore"):
            y = np.asarray(f(grid), dtype=complex)
    except Exception:
        return np.full(grid.size, np.nan)
    y = np.broadcast_to(y, grid.shape).copy()
    out = np.where(np.abs(y.imag) < 1e-9, y.real, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def _rebuild_summary(cv: Path, grid: np.ndarray, manifest: dict) -> pd.DataFrame:
    """The median baseline, recomputed when cv_summary.csv is missing.

    The same comparison build_cv_folds writes: each held-out raw curve, moved into the
    reference frame by its offset, against its fold's training median. The median is read
    from transforms/median_identity.csv -- the curve PySR was fitted to, on the fold grid --
    so the baseline is the one the fits actually competed with.
    """
    offsets = {int(k): float(v) for k, v in manifest["to_ref_offset_dex"].items()}
    rows = []
    for f in manifest["folds"]:
        fold_dir = cv / f["dir"]
        m = pd.read_csv(fold_dir / "transforms" / "median_identity.csv")
        eps, med = np.asarray(m["eps_hat"], float), np.asarray(m["log10f"], float)
        if eps.size != grid.size or np.max(np.abs(eps - grid)) > 1e-9:
            raise SystemExit(f"{fold_dir.name}: median_identity.csv is not on the folds.json "
                             f"grid; cannot rebuild the baseline from it")
        for hid in f["test_halos"]:
            t = pd.read_csv(fold_dir / "test" / f"test_halo{int(hid):02d}.csv")
            resid = np.asarray(t["log10f"], float) + offsets[int(hid)] - med
            mse = float(np.mean(resid ** 2))
            rows.append({"fold": int(f["fold"]), "halo": int(hid),
                         "to_ref_offset_dex": offsets[int(hid)],
                         "mse_vs_train_median": mse})
    out = pd.DataFrame(rows)
    ref = manifest.get("median_baseline_mse_mean")
    if ref is not None and abs(out["mse_vs_train_median"].mean() - float(ref)) > 1e-9:
        raise SystemExit(
            f"rebuilt median baseline {out['mse_vs_train_median'].mean():.6g} disagrees with "
            f"folds.json median_baseline_mse_mean {float(ref):.6g}; the fold files are not "
            f"from the build folds.json describes")
    print(f"cv_summary.csv missing -- median baseline rebuilt from folds.json and "
          f"fold*/transforms/median_identity.csv ({len(out)} held-out halos"
          + (", matches folds.json" if ref is not None else "") + ")")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=str, required=True)
    ap.add_argument("--composed", type=str, default="pysr_composed.csv",
                    help="per-fold composed table to score (default pysr_composed.csv)")
    ap.add_argument("--out", type=str, default="pysr_holdout_scores.csv")
    ap.add_argument("--summary", type=str, default="pysr_holdout_summary.csv")
    args = ap.parse_args()

    cv = Path(args.cv_dir).expanduser().resolve()
    grid, manifest = _fold_grid(cv)
    summary = (pd.read_csv(cv / "cv_summary.csv") if (cv / "cv_summary.csv").exists()
               else _rebuild_summary(cv, grid, manifest))
    offsets = {int(k): float(v) for k, v in manifest["to_ref_offset_dex"].items()}

    # The offset is written in two places by build_cv_folds. If they ever disagree, one of
    # the two files is from a different vintage and every score below is in the wrong frame.
    for _, r in summary.iterrows():
        if abs(float(r["to_ref_offset_dex"]) - offsets[int(r["halo"])]) > 1e-12:
            raise SystemExit(
                f"halo {int(r['halo'])}: cv_summary.csv and folds.json disagree on "
                f"to_ref_offset_dex; rebuild the folds before scoring")

    n_folds = 0
    for fold_dir in sorted(cv.glob("fold*")):
        composed = fold_dir / args.composed
        if not composed.exists():
            continue
        n_folds += 1
        front = pd.read_csv(composed)
        held = summary[summary["fold"] == int(front["fold"].iloc[0])]

        curves = {}
        for _, h in held.iterrows():
            hid = int(h["halo"])
            t = pd.read_csv(fold_dir / "test" / f"test_halo{hid:02d}.csv")
            # raw -> reference frame, the one the training median and the fits live in
            curves[hid] = np.asarray(t["log10f"], float) + offsets[hid]

        rows = []
        for _, e in front.iterrows():
            y = _evaluate(str(e["equation_eps"]), grid)
            for _, h in held.iterrows():
                hid, base = int(h["halo"]), float(h["mse_vs_train_median"])
                resid = y - curves[hid]
                ok = np.isfinite(resid)
                mse = float(np.mean(resid[ok] ** 2)) if ok.any() else np.nan
                rows.append({
                    "fold": int(e["fold"]), "halo": hid, "transform": e["transform"],
                    "complexity_u": int(e["complexity_u"]),
                    "complexity_eps": int(e["complexity_eps"]),
                    "complexity_eps_in_alphabet": e["complexity_eps_in_alphabet"],
                    "train_loss": float(e["loss"]),
                    "n_valid": int(ok.sum()), "n_points": int(grid.size),
                    "mse": mse, "rmse": float(np.sqrt(mse)) if mse == mse else np.nan,
                    "max_abs_resid": (float(np.max(np.abs(resid[ok]))) if ok.any()
                                      else np.nan),
                    "baseline_mse": base,
                    "skill": (1.0 - mse / base) if (mse == mse and base > 0) else np.nan,
                    "equation_eps": e["equation_eps"],
                })
        scores = pd.DataFrame(rows)
        # Ordered by cost, never by score: see the module docstring.
        scores = scores.sort_values(
            ["transform", "complexity_eps", "halo"], kind="mergesort"
        ).reset_index(drop=True)
        scores.to_csv(fold_dir / args.out, index=False)

        agg = (scores.groupby(["transform", "complexity_u", "complexity_eps"], as_index=False)
                     .agg(complexity_eps_in_alphabet=("complexity_eps_in_alphabet", "first"),
                          train_loss=("train_loss", "first"),
                          mean_mse=("mse", "mean"), worst_mse=("mse", "max"),
                          mean_skill=("skill", "mean"),
                          halos_beating_median=("skill", lambda s: int((s > 0).sum())),
                          n_halos=("skill", "size"),
                          min_valid=("n_valid", "min"),
                          equation_eps=("equation_eps", "first")))
        agg = agg.sort_values(["transform", "complexity_eps"],
                              kind="mergesort").reset_index(drop=True)
        agg.to_csv(fold_dir / args.summary, index=False)

        j = int(front["fold"].iloc[0])
        print(f"\nfold {j}: {len(front)} expressions x {len(held)} held-out halos "
              f"-> {fold_dir / args.out}")
        print(f"  median baseline over these halos: "
              f"{held['mse_vs_train_median'].mean():.4f} dex^2 mean, "
              f"{held['mse_vs_train_median'].min():.4f}-"
              f"{held['mse_vs_train_median'].max():.4f} range")
        beat = agg[agg["halos_beating_median"] == agg["n_halos"]]
        partial = agg[(agg["halos_beating_median"] > 0)
                      & (agg["halos_beating_median"] < agg["n_halos"])]
        print(f"  {len(beat)} of {len(agg)} expressions beat the median on all "
              f"{int(agg['n_halos'].iloc[0])} halos, {len(partial)} on some")
        incomplete = agg[agg["min_valid"] < grid.size]
        if len(incomplete):
            print(f"  {len(incomplete)} are not defined on the whole grid "
                  f"(min valid points {int(incomplete['min_valid'].min())}/{grid.size})")

    if not n_folds:
        raise FileNotFoundError(
            f"no {args.composed} under {cv}/fold*/ -- run compose_pysr_fronts first")
    print(f"\nscored {n_folds} folds. These are out-of-sample numbers for forms already "
          f"chosen;\nchoosing a form by reading down the mse column spends the only "
          f"evidence left that is\nindependent of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
