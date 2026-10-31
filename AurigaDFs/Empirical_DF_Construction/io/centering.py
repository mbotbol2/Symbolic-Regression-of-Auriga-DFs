from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py


def _read_first_row_from_any_piece(files: list[Path], dset_path: str) -> np.ndarray | None:
    for fn in files:
        try:
            with h5py.File(fn, "r") as f:
                if dset_path in f:
                    return np.asarray(f[dset_path][0], dtype=float)
        except OSError:
            continue
    return None


def load_center_from_catalog(
    *,
    groupcat_files_list: list[Path],
    center_mode: str,
    a: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns:
      - center_pos_raw: comoving Mpc/h (same unit system as snapshot Coordinates)
      - center_vel_kms: km/s, already converted per convention
      - center_src: 'subhalo0' or 'group0'

    Conventions:
      - SubhaloVel is km/s already
      - GroupVel must be divided by a (1/a scaling) before comparing/subtracting in km/s
    """
    mode = str(center_mode).lower().strip()
    if mode not in {"subhalo", "group"}:
        raise ValueError("center_mode must be 'subhalo' or 'group'")

    if mode == "subhalo":
        p = _read_first_row_from_any_piece(groupcat_files_list, "Subhalo/SubhaloPos")
        v = _read_first_row_from_any_piece(groupcat_files_list, "Subhalo/SubhaloVel")
        if p is None or v is None:
            raise FileNotFoundError("Missing SubhaloPos/SubhaloVel in group catalog files")
        return np.asarray(p, float), np.asarray(v, float), "subhalo0"

    p = _read_first_row_from_any_piece(groupcat_files_list, "Group/GroupPos")
    v = _read_first_row_from_any_piece(groupcat_files_list, "Group/GroupVel")
    if p is None or v is None:
        raise FileNotFoundError("Missing GroupPos/GroupVel in group catalog files")
    v = np.asarray(v, float)
    if np.isfinite(a) and float(a) != 0.0:
        v = v / float(a)
    return np.asarray(p, float), v, "group0"

