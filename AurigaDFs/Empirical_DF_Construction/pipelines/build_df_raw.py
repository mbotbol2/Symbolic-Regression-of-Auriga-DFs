"""Isotropic DF f(E) per halo from its bound particles (build_ids first), with the potential
from the particles out to r_max."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..core.df_from_particles import df_isotropic_from_particles_rmax
from ..core.df_math import default_eps_grid
from ..io.auriga_root import resolve_auriga_data_root
from ..io.layout import resolve_halo_dir
from ..io.results_paths import df_raw_dir, ids_dir
from ..io.subset import load_subset_phase_space_at_snap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", required=True)
    ap.add_argument("--method", default="with_hubble_flow", choices=["with_hubble_flow", "without_hubble_flow"])
    ap.add_argument("--snap", type=int, default=127)
    ap.add_argument("--pt", type=int, default=1)
    ap.add_argument("--center-mode", type=str, default="subhalo", choices=["subhalo", "group"])
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--phi-zero-at", type=str, default="infty", choices=["infty", "rmax"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = resolve_auriga_data_root()
    eps_hat = default_eps_grid()

    for hid in [int(x) for x in args.halos]:
        out_df = df_raw_dir(hid)
        out_df.mkdir(parents=True, exist_ok=True)

        csv_path = out_df / f"df_raw_{args.method}.csv"
        meta_path = out_df / f"df_raw_{args.method}_meta.json"
        png_path = out_df / f"df_raw_{args.method}.png"

        if csv_path.exists() and meta_path.exists() and png_path.exists() and (not bool(args.force)):
            print(f"[halo {hid}] skip DF (exists)")
            continue

        ids_root = ids_dir(hid)
        ids_path = ids_root / f"bound_ids_{args.method}_{str(args.center_mode)}.npy"
        bound_ids = np.asarray(np.load(ids_path), dtype=np.uint64)
        bound_ids.sort()
        print(f"[halo {hid}] loading subset for DF (n={bound_ids.size})...")

        hdir = resolve_halo_dir(Path(data_root), hid)
        dat = load_subset_phase_space_at_snap(
            halo_dir=hdir,
            subset_ids_sorted=bound_ids,
            snap=int(args.snap),
            pt=int(args.pt),
            center_mode=str(args.center_mode),
            apply_vel_sqrt_a=True,
            chunk=int(args.chunk),
        )

        eps, log10f, meta = df_isotropic_from_particles_rmax(
            x_kpc=dat["x_kpc"],
            v_kms=dat["v_kms"],
            m_msun=dat["m_msun"],
            eps_hat=eps_hat,
            phi_zero_at=str(args.phi_zero_at),
        )

        # write CSV
        with csv_path.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["eps_hat", "log10f"])
            for ee, lf in zip(eps.tolist(), log10f.tolist()):
                w.writerow([float(ee), float(lf) if np.isfinite(lf) else "nan"])

        meta2 = dict(meta)
        meta2.update(
            {
                "halo": hid,
                "method": args.method,
                "center_mode": str(args.center_mode),
                "snap": int(args.snap),
                "pt": int(args.pt),
            }
        )
        meta_path.write_text(json.dumps(meta2, indent=2, sort_keys=True))

        # plot
        plt.figure(figsize=(6.6, 4.8))
        m = np.isfinite(log10f)
        plt.plot(eps[m], log10f[m], lw=1.6)
        plt.xlabel(r"$\hat{\epsilon}=E/\Psi_0$")
        plt.ylabel(r"$\log_{10} f(E)$")
        plt.title(f"Halo {hid:02d} DF ({args.method.replace('_', ' ')}; outer=rmax, Phi({args.phi_zero_at})=0)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

        print(f"[halo {hid}] wrote {csv_path.name}, {png_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

