"""Bound DM particle IDs per halo: first pericentre then first apocentre between snap_min and
snap_max, with or without the Hubble flow in the radial velocity (see bound_selection)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..bound_selection.common import BoundSelectionConfig, compute_bound_ids_by_signflip
from ..io.auriga_root import resolve_auriga_data_root
from ..io.id_cache import build_dm_id_cache, ensure_sorted_unique_uint64
from ..io.layout import resolve_halo_dir
from ..io.results_paths import ids_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--halos", type=int, nargs="+", required=True, help="Halo IDs to process")
    ap.add_argument("--method", default="with_hubble_flow", choices=["with_hubble_flow", "without_hubble_flow"])
    ap.add_argument("--center-mode", type=str, default="subhalo", choices=["subhalo", "group"])
    ap.add_argument("--snap-min", type=int, default=77)
    ap.add_argument("--snap-max", type=int, default=127)
    ap.add_argument("--pt", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data_root = resolve_auriga_data_root()

    for hid in [int(x) for x in args.halos]:
        hdir = resolve_halo_dir(Path(data_root), hid)
        out_ids = ids_dir(hid)
        out_ids.mkdir(parents=True, exist_ok=True)

        ids0_path = out_ids / f"dm_ids_snap{int(args.snap_min):03d}.npy"
        if ids0_path.exists() and (not bool(args.force)):
            ids0_sorted = np.asarray(np.load(ids0_path), dtype=np.uint64)
            ids0_sorted = ensure_sorted_unique_uint64(ids0_sorted)
        else:
            ids0_sorted = build_dm_id_cache(
                halo_dir=hdir,
                snap=int(args.snap_min),
                out_path=ids0_path,
                pt=int(args.pt),
                chunk=int(args.chunk),
            )

        cfg = BoundSelectionConfig(
            snap_min=int(args.snap_min),
            snap_max=int(args.snap_max),
            pt=int(args.pt),
            center_mode=str(args.center_mode),
            apply_vel_sqrt_a=True,
            chunk=int(args.chunk),
            eps_vr=0.0,
        )

        ids_path = out_ids / f"bound_ids_{args.method}_{str(args.center_mode)}.npy"
        r_apo_path = out_ids / f"r_apo_first_kpc_{args.method}_{str(args.center_mode)}.npy"
        meta_path = out_ids / f"bound_{args.method}_meta.json"

        if ids_path.exists() and r_apo_path.exists() and meta_path.exists() and (not bool(args.force)):
            print(f"[halo {hid}] skip (exists)")
            continue

        print(f"[halo {hid}] computing bound IDs ({args.method.replace('_', ' ')})...")
        mask, r_apo_first = compute_bound_ids_by_signflip(
            halo_dir=hdir, ids0_sorted=ids0_sorted, cfg=cfg, hubble_flow=args.method == "with_hubble_flow")
        bound_ids = np.asarray(ids0_sorted[mask], dtype=np.uint64)
        bound_ids = ensure_sorted_unique_uint64(bound_ids)

        np.save(ids_path, bound_ids)
        np.save(r_apo_path, np.asarray(r_apo_first, np.float32))
        meta_path.write_text(
            json.dumps(
                {
                    "halo": hid,
                    "center_mode": str(args.center_mode),
                    "pt": int(args.pt),
                    "snap_min": int(args.snap_min),
                    "snap_max": int(args.snap_max),
                    "n_total": int(ids0_sorted.size),
                    "n_bound": int(bound_ids.size),
                },
                indent=2,
                sort_keys=True,
            )
        )
        print(f"[halo {hid}] wrote {ids_path.name} (n={bound_ids.size})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

