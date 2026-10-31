"""
Block hold-out cross-validation folds for the DF.

The halos are split into consecutive blocks of --holdout (30 halos, 5 per block: 6 folds).
Each block is a test set; its training curve is the median of the other halos, in the
reference halo's frame (Halo 2). Test curves are written raw, in each halo's own units, with
the per-halo constant that maps them across in folds.json:

    log10 f_{i->ref} = log10 f_raw,i + to_ref_offset_dex[i],
    to_ref_offset_dex[i] = log10 f_s,i - log10 f_s,ref,   f_s = G^{3/2} r_s^3 rho_s^{1/2}

The abscissa is eps_hat = E/Psi_i(0) throughout, on one grid shared by every fold (the range
all halos cover), so errors are comparable across folds.

Reads   Halo_XX/df/raw/df_raw_<method>.csv, Halo_XX/df/scaled/df_ref<NN>_<method>_<tag>.csv
        and its _meta.json
Writes  <out>/fold<j>/... training median and raw test curves, <out>/cv_summary.csv (each
        held-out halo's error against its fold's median: the baseline), <out>/folds.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..core.df_scaling import df_scale
from ..core.df_transforms import TRANSFORM_NAMES, apply as apply_transform
from ..io.results_paths import df_dir, df_raw_dir, sr_dir


def _read_named(path: Path, abscissa: str, ordinates: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Read a DF CSV by header name. Never by column index: the reference-scaled files are
    eps_hat, eps_hat_ref, log10f_ref, so column 1 is a second abscissa, not the DF."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    names = set(d.dtype.names or ())
    if abscissa not in names:
        raise RuntimeError(f"{path}: no {abscissa} column (found {sorted(names)})")
    ordinate = next((c for c in ordinates if c in names), None)
    if ordinate is None:
        raise RuntimeError(f"{path}: none of {ordinates} present (found {sorted(names)})")
    return np.asarray(d[abscissa], float), np.asarray(d[ordinate], float)


def _load_raw(hid: int, method: str) -> tuple[np.ndarray, np.ndarray] | None:
    p = df_raw_dir(hid) / f"df_raw_{method}.csv"
    if not p.exists():
        return None
    return _read_named(p, "eps_hat", ("log10f",))


def _load_ref(hid: int, method: str, tag: str, ref_hid: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Reference-scaled curve against eps_hat (the halo's own), not eps_hat_ref."""
    p = df_dir(hid) / "scaled" / f"df_ref{ref_hid:02d}_{method}_{tag}.csv"
    if not p.exists():
        return None
    return _read_named(p, "eps_hat", ("log10f_ref",))


def _load_scales(hid: int, method: str, tag: str) -> tuple[float, float] | None:
    """(length scale, density scale) the scaling pipeline actually used for this halo."""
    p = df_dir(hid) / "scaled" / f"df_scaled_{method}_{tag}_meta.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    L, rho = m.get("length_scale_kpc"), m.get("density_scale_msun_kpc3")
    if L is None or rho is None:
        return None
    return float(L), float(rho)


def _blocks(halos: list[int], k: int) -> list[list[int]]:
    """Consecutive blocks of k, over the halos actually present. A short final block is kept."""
    return [halos[i:i + k] for i in range(0, len(halos), k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--method", type=str, default="with_hubble_flow",
                    choices=["with_hubble_flow", "without_hubble_flow"])
    ap.add_argument("--scale-by", type=str, default="rs", choices=["rs", "virial"])
    ap.add_argument("--shape", type=str, default="nfw", choices=["nfw", "einasto"])
    ap.add_argument("--ref-halo", type=int, default=2,
                    help="frame the training median is expressed in")
    ap.add_argument("--holdout", type=int, default=5, help="halos held out per fold")
    ap.add_argument("--eps-lo", type=float, default=0.03)
    ap.add_argument("--eps-hi", type=float, default=0.999)
    ap.add_argument("--n-grid", type=int, default=400)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="default: <Halo_Results>/_sr/cv_<method>_<tag>")
    args = ap.parse_args()

    method, shape, scale_by = str(args.method), str(args.shape), str(args.scale_by)
    ref_hid = int(args.ref_halo)
    tag = shape if scale_by == "rs" else "virial"
    eps_lo, eps_hi = float(args.eps_lo), float(args.eps_hi)
    k = int(args.holdout)
    if k < 1:
        raise ValueError(f"--holdout must be >= 1, got {k}")
    if not (0.0 < eps_lo < eps_hi < 1.0):
        raise ValueError(f"Need 0 < eps_lo < eps_hi < 1, got [{eps_lo}, {eps_hi}]")

    out = Path(args.out_dir) if args.out_dir else sr_dir() / f"cv_{method}_{tag}_ref{ref_hid:02d}"
    out.mkdir(parents=True, exist_ok=True)

    # ---- reference-frame scales, for the raw -> halo-<ref> offset ---------------------
    ref_scales = _load_scales(ref_hid, method, tag)
    if ref_scales is None:
        raise RuntimeError(
            f"no df_scaled_{method}_{tag}_meta.json for the reference halo {ref_hid}; "
            f"run build_df_dimensionless --ref-halo {ref_hid} first")
    log10_fs_ref = float(np.log10(df_scale(ref_scales[0], ref_scales[1])))

    # ---- load raw + reference-scaled, window, keep only halos with both ---------------
    raw: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ref: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    offset: dict[int, float] = {}
    missing: list[int] = []

    def _window(xy):
        x, y = xy
        m = np.isfinite(x) & np.isfinite(y) & (x >= eps_lo) & (x <= eps_hi)
        if np.count_nonzero(m) < 2:
            return None
        o = np.argsort(x[m])
        return x[m][o], y[m][o]

    for hid in sorted(set(int(h) for h in args.halos)):
        g_raw = _load_raw(hid, method)
        g_ref = _load_ref(hid, method, tag, ref_hid)
        sc = _load_scales(hid, method, tag)
        if g_raw is None or g_ref is None or sc is None:
            missing.append(hid)
            continue
        w_raw, w_ref = _window(g_raw), _window(g_ref)
        if w_raw is None or w_ref is None:
            missing.append(hid)
            continue
        raw[hid], ref[hid] = w_raw, w_ref
        offset[hid] = float(np.log10(df_scale(sc[0], sc[1]))) - log10_fs_ref
    if missing:
        print(f"skipped (missing raw, ref{ref_hid:02d} or scale meta): {missing}")
    halos = sorted(raw)
    if len(halos) < k + 1:
        raise RuntimeError(f"need more than --holdout={k} halos to leave a training set, got {len(halos)}")

    # ---- one grid for every fold, shared by both frames -------------------------------
    lo = max(max(float(raw[h][0][0]), float(ref[h][0][0])) for h in halos)
    hi = min(min(float(raw[h][0][-1]), float(ref[h][0][-1])) for h in halos)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise RuntimeError(f"halos share no common abscissa range: [{lo}, {hi}]")
    grid = np.linspace(lo, hi, int(args.n_grid))
    stack_ref = np.array([np.interp(grid, ref[h][0], ref[h][1]) for h in halos])
    stack_raw = np.array([np.interp(grid, raw[h][0], raw[h][1]) for h in halos])
    print(f"{len(halos)} halos on a shared grid of {grid.size} points, "
          f"eps_hat in [{lo:.4f}, {hi:.4f}]")
    print(f"training median in halo-{ref_hid:02d} frame; held-out curves written raw")

    # The offset must reproduce the stored reference curve from the raw one. Checking it
    # here catches a raw/scaled pair rebuilt at different vintages, which would otherwise
    # show up only as a mysteriously large test error.
    for i, h in enumerate(halos):
        d = float(np.max(np.abs((stack_raw[i] + offset[h]) - stack_ref[i])))
        if d > 1e-6:
            raise RuntimeError(
                f"halo {h}: raw + offset does not reproduce df_ref{ref_hid:02d} "
                f"(max |diff| = {d:.3e} dex). Rebuild raw and scaled together with --force.")
    print("offset check: raw + to_ref_offset_dex reproduces every stored reference curve")

    folds = _blocks(halos, k)
    print(f"{len(folds)} folds of {k} held out: " + ", ".join("{" + ",".join(map(str, b)) + "}" for b in folds))

    idx_of = {h: i for i, h in enumerate(halos)}
    summary_rows = []
    manifest_folds = []

    for j, test_halos in enumerate(folds, start=1):
        train_halos = [h for h in halos if h not in set(test_halos)]
        tr = stack_ref[[idx_of[h] for h in train_halos]]      # median in the halo-<ref> frame
        med = np.median(tr, axis=0)
        p16 = np.percentile(tr, 16, axis=0)
        p84 = np.percentile(tr, 84, axis=0)

        fold_dir = out / f"fold{j}"
        (fold_dir / "transforms").mkdir(parents=True, exist_ok=True)
        (fold_dir / "test").mkdir(parents=True, exist_ok=True)

        train_csv = fold_dir / "train_median.csv"
        with train_csv.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["eps_hat", "median", "p16", "p84"])
            for a, b, c, d_ in zip(grid, med, p16, p84):
                w.writerow([float(a), float(b), float(c), float(d_)])

        # One CSV per change of variable, in the column order PySR is fed:
        # transformed_var first, so col 0 vs col 1 is u vs log10 f.
        n_written = 0
        for name in TRANSFORM_NAMES:
            u = apply_transform(name, grid)
            ok = np.isfinite(u) & np.isfinite(med)
            if np.count_nonzero(ok) < 20:
                print(f"[fold {j}] transform {name}: only {np.count_nonzero(ok)} finite points, skipped")
                continue
            tp = fold_dir / "transforms" / f"median_{name}.csv"
            with tp.open("w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["transformed_var", "log10f", "eps_hat", "p16", "p84"])
                for a, b, c, d_, e_ in zip(u[ok], med[ok], grid[ok], p16[ok], p84[ok]):
                    w.writerow([float(a), float(b), float(c), float(d_), float(e_)])
            n_written += 1

        for h in test_halos:
            y_raw = stack_raw[idx_of[h]]                       # written RAW, halo's own units
            with (fold_dir / "test" / f"test_halo{h:02d}.csv").open("w", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(["eps_hat", "log10f"])
                for a, b in zip(grid, y_raw):
                    w.writerow([float(a), float(b)])
            # score in the reference frame, the way a test script would
            resid = (y_raw + offset[h]) - med
            mse = float(np.mean(resid**2))
            summary_rows.append({
                "fold": j, "halo": h, "n_train": len(train_halos), "n_points": int(grid.size),
                "to_ref_offset_dex": offset[h],
                "mse_vs_train_median": mse,
                "rmse_vs_train_median": float(np.sqrt(mse)),
                "max_abs_resid": float(np.max(np.abs(resid))),
            })

        manifest_folds.append({
            "fold": j, "test_halos": test_halos, "train_halos": train_halos,
            "dir": fold_dir.name,
            "train_median_csv": f"{fold_dir.name}/train_median.csv",
            "transforms": [f"{fold_dir.name}/transforms/median_{n}.csv" for n in TRANSFORM_NAMES],
            "test_csvs": [f"{fold_dir.name}/test/test_halo{h:02d}.csv" for h in test_halos],
        })
        print(f"[fold {j}] train {len(train_halos)} halos, test {test_halos} "
              f"-> {fold_dir.name}/ ({n_written} transforms, {len(test_halos)} test CSVs)")

    summary_csv = out / "cv_summary.csv"
    fields = ["fold", "halo", "n_train", "n_points", "to_ref_offset_dex",
              "mse_vs_train_median", "rmse_vs_train_median", "max_abs_resid"]
    with summary_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    all_mse = np.array([r["mse_vs_train_median"] for r in summary_rows], float)
    (out / "folds.json").write_text(json.dumps({
        "method": method, "scale_by": scale_by, "shape": shape, "tag": tag,
        "ref_halo": ref_hid,
        "median_frame": f"halo_{ref_hid:02d}",
        "test_curve_frame": "raw",
        "abscissa": "eps_hat",
        "to_ref_offset_dex": {str(h): offset[h] for h in halos},
        "offset_definition": "log10 f_{i->ref} = log10 f_raw,i + to_ref_offset_dex[i]",
        "transforms": TRANSFORM_NAMES,
        "holdout": k, "n_folds": len(folds), "halos": halos, "skipped": missing,
        "eps_lo": eps_lo, "eps_hi": eps_hi,
        "grid_lo": float(lo), "grid_hi": float(hi), "n_grid": int(grid.size),
        "source_raw": f"Halo_XX/df/raw/df_raw_{method}.csv",
        "source_ref": f"Halo_XX/df/scaled/df_ref{ref_hid:02d}_{method}_{tag}.csv",
        "folds": manifest_folds,
        "median_baseline_mse_mean": float(np.mean(all_mse)),
    }, indent=2))

    print(f"wrote {summary_csv}")
    print(f"median-baseline MSE over all {len(summary_rows)} held-out halos: "
          f"mean {np.mean(all_mse):.4f}, median {np.median(all_mse):.4f} dex^2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
