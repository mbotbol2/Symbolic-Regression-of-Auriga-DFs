from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..core.density_fits import einasto_shape, nfw_shape, fit_nfw_einasto_gridsearch
from ..core.density_profile import density_profile_from_positions
from ..io.auriga_root import resolve_auriga_data_root
from ..io.layout import resolve_halo_dir
from ..io.results_paths import density_dir, ids_dir
from ..io.subset import load_subset_phase_space_at_snap


def _load_ids(ids_root: Path, method: str, center_mode: str) -> np.ndarray:
    method = str(method).strip().lower()
    if method == "without_hubble_flow":
        fn = ids_root / f"bound_ids_without_hubble_flow_{center_mode}.npy"
    elif method == "with_hubble_flow":
        fn = ids_root / f"bound_ids_with_hubble_flow_{center_mode}.npy"
    else:
        raise ValueError("method must be 'without_hubble_flow' or 'with_hubble_flow'")
    return np.asarray(np.load(fn), dtype=np.uint64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", required=True)
    ap.add_argument("--method", type=str, required=True, choices=["without_hubble_flow", "with_hubble_flow"])
    ap.add_argument("--snap", type=int, default=127)
    ap.add_argument("--pt", type=int, default=1)
    ap.add_argument("--center-mode", type=str, default="subhalo", choices=["subhalo", "group"])
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--nbins", type=int, default=90)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = resolve_auriga_data_root()

    for hid in [int(x) for x in args.halos]:
        out_den = density_dir(hid)
        out_den.mkdir(parents=True, exist_ok=True)

        csv_path = out_den / f"density_{args.method}.csv"
        fit_path = out_den / f"density_fit_params_{args.method}.json"
        png_path = out_den / f"density_{args.method}.png"

        if csv_path.exists() and fit_path.exists() and png_path.exists() and (not bool(args.force)):
            print(f"[halo {hid}] skip density (exists)")
            continue

        ids_root = ids_dir(hid)
        bound_ids = _load_ids(ids_root, str(args.method), str(args.center_mode))
        print(f"[halo {hid}] loading subset for density (n={bound_ids.size})...")

        hdir = resolve_halo_dir(Path(data_root), hid)
        dat = load_subset_phase_space_at_snap(
            halo_dir=hdir,
            subset_ids_sorted=np.sort(bound_ids),
            snap=int(args.snap),
            pt=int(args.pt),
            center_mode=str(args.center_mode),
            apply_vel_sqrt_a=True,
            chunk=int(args.chunk),
        )
        x = dat["x_kpc"]
        m = dat["m_msun"]

        r_mid, rho, counts = density_profile_from_positions(x_kpc=x, m_msun=m, nbins=int(args.nbins))
        fit = fit_nfw_einasto_gridsearch(r_mid, rho, counts)

        # write CSV
        with csv_path.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["r_kpc", "rho_msun_kpc3", "counts"])
            for rr, rh, cc in zip(r_mid.tolist(), rho.tolist(), counts.tolist()):
                w.writerow([float(rr), float(rh), int(cc)])
        fit_path.write_text(json.dumps(fit, indent=2, sort_keys=True))

        # plot
        plt.figure(figsize=(6.6, 4.8))
        plt.loglog(r_mid, rho, lw=1.5, label=f"data ({args.method})")
        if fit and "nfw" in fit:
            rs = float(fit["nfw"]["rs"])
            rhos = float(fit["nfw"]["rhos"])
            plt.loglog(r_mid, rhos * nfw_shape(r_mid, rs), ls="--", lw=1.2, label="NFW fit (shape)")
        if fit and "einasto" in fit:
            r2 = float(fit["einasto"]["r2"])
            alpha = float(fit["einasto"]["alpha"])
            rho2 = float(fit["einasto"]["rho2"])
            plt.loglog(r_mid, rho2 * einasto_shape(r_mid, r2, alpha), ls=":", lw=1.3, label="Einasto fit (shape)")
        plt.xlabel("r  [kpc]")
        plt.ylabel(r"$\rho(r)$  [M$_\odot$ / kpc$^3$]")
        plt.title(f"Halo {hid:02d} density ({args.method})")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

        print(f"[halo {hid}] wrote {csv_path.name}, {png_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

