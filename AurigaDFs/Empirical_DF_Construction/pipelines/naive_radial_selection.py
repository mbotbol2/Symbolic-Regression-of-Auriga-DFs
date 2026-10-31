"""
Naive radial selection: DF from all particles within R200m of central subhalo.

No orbit-based selection; purely radial cut at the target snapshot.
"""

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
from ..io.radial import load_particles_within_R200m
from ..io.results_paths import df_raw_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="DF from all particles within R200m (naive radial selection)")
    ap.add_argument("--halos", type=int, nargs="+", required=True)
    ap.add_argument("--snap", type=int, default=127)
    ap.add_argument("--pt", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--phi-zero-at", type=str, default="infty", choices=["infty", "rmax"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = resolve_auriga_data_root()
    eps_hat = default_eps_grid()

    for hid in [int(x) for x in args.halos]:
        out_df = df_raw_dir(hid)
        out_df.mkdir(parents=True, exist_ok=True)

        csv_path = out_df / "df_naive_radial_R200m.csv"
        meta_path = out_df / "df_naive_radial_R200m_meta.json"
        png_path = out_df / "df_naive_radial_R200m.png"

        if csv_path.exists() and meta_path.exists() and png_path.exists() and (not bool(args.force)):
            print(f"[halo {hid}] skip naive radial DF (exists)")
            continue

        hdir = resolve_halo_dir(Path(data_root), hid)
        print(f"[halo {hid}] loading particles within R200m...")
        dat = load_particles_within_R200m(
            halo_dir=hdir,
            snap=int(args.snap),
            pt=int(args.pt),
            center_mode="subhalo",
            apply_vel_sqrt_a=True,
            chunk=int(args.chunk),
        )
        print(f"[halo {hid}] n={dat['x_kpc'].shape[0]:,} R200m={float(dat['R200m_kpc']):.1f} kpc")

        eps, log10f, meta = df_isotropic_from_particles_rmax(
            x_kpc=dat["x_kpc"],
            v_kms=dat["v_kms"],
            m_msun=dat["m_msun"],
            eps_hat=eps_hat,
            phi_zero_at=str(args.phi_zero_at),
        )

        with csv_path.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["eps_hat", "log10f"])
            for ee, lf in zip(eps.tolist(), log10f.tolist()):
                w.writerow([float(ee), float(lf) if np.isfinite(lf) else "nan"])

        meta2 = dict(meta)
        meta2.update(
            {
                "halo": hid,
                "method": "naive_radial_R200m",
                "center_mode": "subhalo",
                "snap": int(args.snap),
                "pt": int(args.pt),
                "R200m_kpc": float(dat["R200m_kpc"]),
                "n_particles": int(dat["x_kpc"].shape[0]),
            }
        )
        meta_path.write_text(json.dumps(meta2, indent=2, sort_keys=True))

        plt.figure(figsize=(6.6, 4.8))
        m = np.isfinite(log10f)
        plt.plot(eps[m], log10f[m], lw=1.6)
        plt.xlabel(r"$\hat{\epsilon}=E/\Psi_0$")
        plt.ylabel(r"$\log_{10} f(E)$")
        plt.title(f"Halo {hid:02d} DF (naive radial, r<R200m; Phi({args.phi_zero_at})=0)")
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

        print(f"[halo {hid}] wrote {csv_path.name}, {png_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
