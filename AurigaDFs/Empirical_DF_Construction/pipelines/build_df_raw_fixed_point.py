from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ..core.df_fixed_point import df_fixed_point_from_particles
from ..io.auriga_root import resolve_auriga_data_root
from ..io.id_cache import build_dm_id_cache, ensure_sorted_unique_uint64
from ..io.layout import resolve_halo_dir
from ..io.results_paths import df_raw_dir, ids_dir
from ..io.subset import load_subset_phase_space_at_snap


def _to_jsonable(x):
    """
    Convert numpy types (ndarray / scalars) into JSON-serialisable Python objects.
    Keeps nested dict/list structures.
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # last resort (should be rare)
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", required=True)
    ap.add_argument("--snap", type=int, default=127)
    ap.add_argument("--pt", type=int, default=1)
    ap.add_argument("--center-mode", type=str, default="subhalo", choices=["subhalo", "group"])
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = resolve_auriga_data_root()

    for hid in [int(x) for x in args.halos]:
        out_df = df_raw_dir(hid)
        out_df.mkdir(parents=True, exist_ok=True)

        csv_path = out_df / "df_raw_fixed_point.csv"
        meta_path = out_df / "df_raw_fixed_point_meta.json"
        png_path = out_df / "df_raw_fixed_point.png"

        if csv_path.exists() and meta_path.exists() and png_path.exists() and (not bool(args.force)):
            print(f"[halo {hid}] skip fixed-point DF (exists)")
            continue

        # Use *all* DM particles at this snapshot (fixed-point method decides its own region).
        out_ids = ids_dir(hid)
        out_ids.mkdir(parents=True, exist_ok=True)
        ids0_path = out_ids / f"dm_ids_snap{int(args.snap):03d}.npy"
        if ids0_path.exists() and (not bool(args.force)):
            ids0 = np.asarray(np.load(ids0_path), dtype=np.uint64)
            ids0 = ensure_sorted_unique_uint64(ids0)
        else:
            hdir = resolve_halo_dir(Path(data_root), hid)
            ids0 = build_dm_id_cache(
                halo_dir=hdir,
                snap=int(args.snap),
                out_path=ids0_path,
                pt=int(args.pt),
                chunk=int(args.chunk),
            )

        print(f"[halo {hid}] loading ALL DM for fixed-point DF (n={ids0.size})...")
        hdir = resolve_halo_dir(Path(data_root), hid)
        dat = load_subset_phase_space_at_snap(
            halo_dir=hdir,
            subset_ids_sorted=ids0,
            snap=int(args.snap),
            pt=int(args.pt),
            center_mode=str(args.center_mode),
            apply_vel_sqrt_a=True,
            chunk=int(args.chunk),
        )

        res = df_fixed_point_from_particles(
            x_kpc=dat["x_kpc"],
            v_kms=dat["v_kms"],
            m_msun=dat["m_msun"],
        )

        eps = np.asarray(res["eps_hat"], float)
        log10f = np.asarray(res["log10f"], float)

        with csv_path.open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["eps_hat", "log10f"])
            for ee, lf in zip(eps.tolist(), log10f.tolist()):
                w.writerow([float(ee), float(lf) if np.isfinite(lf) else "nan"])

        meta = dict(res)
        # trim big arrays from meta json
        for k in ["fE", "gE", "dMdE", "mean_vr", "frac_vr_pos", "log10f_eddington_from_density"]:
            if k in meta:
                meta.pop(k, None)
        meta.update(
            {
                "halo": hid,
                "snap": int(args.snap),
                "pt": int(args.pt),
                "center_mode": str(args.center_mode),
                "df_potential_outer_boundary": "rmax_bound",
                "df_potential_zero_at": "infty",
                "mass_normalised": False,
            }
        )
        meta_path.write_text(json.dumps(_to_jsonable(meta), indent=2, sort_keys=True))

        plt.figure(figsize=(6.6, 4.8))
        m = np.isfinite(log10f)
        plt.plot(eps[m], log10f[m], lw=1.6)
        plt.xlabel(r"$\hat{\epsilon}=E/\Psi_0$")
        plt.ylabel(r"$\log_{10} f(E)$")
        plt.title("Halo %02d DF (fixed-point density; Phi(infty)=0)" % hid)
        plt.tight_layout()
        plt.savefig(png_path, dpi=180)
        plt.close()

        print(f"[halo {hid}] wrote {csv_path.name}, {png_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

