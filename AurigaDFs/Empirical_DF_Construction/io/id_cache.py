from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

from .layout import resolve_data_dir_for_snap
from .files import snapshot_files


def ensure_sorted_unique_uint64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint64)
    x.sort()
    if x.size == 0:
        return x
    return np.unique(x)


def build_dm_id_cache(
    *,
    halo_dir: Path,
    snap: int,
    out_path: Path,
    pt: int = 1,
    chunk: int = 2_000_000,
) -> np.ndarray:
    """
    Cache sorted unique DM particle IDs for this halo at a given snapshot.
    """
    halo_dir = Path(halo_dir).expanduser().resolve()
    snap = int(snap)
    pt = int(pt)
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir_for_snap(halo_dir, snap)
    sfiles = snapshot_files(data_dir, snap)
    if not sfiles:
        raise FileNotFoundError(f"No snapshot files found for halo={halo_dir} snap={snap}")

    ids_all: list[np.ndarray] = []
    dset = f"PartType{pt}/ParticleIDs"
    for fn in sfiles:
        with h5py.File(fn, "r") as f:
            if dset not in f:
                continue
            ids = f[dset]
            n = int(ids.shape[0])
            for start in range(0, n, int(chunk)):
                stop = min(start + int(chunk), n)
                ids_all.append(np.asarray(ids[start:stop], dtype=np.uint64))

    if not ids_all:
        raise RuntimeError(f"Could not read any IDs from {sfiles[0]}")

    ids0 = np.concatenate(ids_all, axis=0)
    ids0 = ensure_sorted_unique_uint64(ids0)
    np.save(out_path, ids0)
    return ids0

