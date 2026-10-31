from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py


def _read_scalar_from_any_piece(files: list[Path], dset_path: str) -> float | None:
    for fn in files:
        try:
            with h5py.File(fn, "r") as f:
                if dset_path in f:
                    x = np.asarray(f[dset_path][0], dtype=float)
                    if x.ndim == 0:
                        return float(x)
                    if x.ndim == 1 and x.size >= 1:
                        return float(x.flat[0])
        except OSError:
            continue
    return None


def load_Rvir_kpc(*, gfiles: list[Path], a: float, h: float) -> tuple[float, str]:
    """
    Return (Rvir_kpc, source_dset) where Rvir is interpreted as a 200-overdensity radius.
    Preference order: Crit200 (R200c), then Mean200 (R200m).

    Group catalog radii are in comoving Mpc/h; convert to physical kpc via *1000*a/h.
    """
    a = float(a)
    h = float(h)
    cand = [
        "Group/Group_R_Crit200",
        "Group/Group_R_Mean200",
        "Group/Group_R_TopHat200",
    ]
    for dset in cand:
        x = _read_scalar_from_any_piece(gfiles, dset)
        if x is None:
            continue
        R_kpc = float(x) * 1000.0 * a / h  # Mpc/h -> kpc physical
        if np.isfinite(R_kpc) and R_kpc > 0:
            return R_kpc, dset
    raise FileNotFoundError("Could not find any R200 radius in group catalogs (Crit200/Mean200/TopHat200).")


def load_R200m_kpc(*, gfiles: list[Path], a: float, h: float) -> float:
    """
    Return R200m (mean 200-overdensity radius) in physical kpc.
    Reads Group_R_Mean200 from group catalog; comoving Mpc/h -> physical kpc via *1000*a/h.
    """
    x = _read_scalar_from_any_piece(gfiles, "Group/Group_R_Mean200")
    if x is None:
        raise FileNotFoundError("Group/Group_R_Mean200 not found in group catalogs")
    R_kpc = float(x) * 1000.0 * float(a) / float(h)  # Mpc/h -> kpc physical
    if not (np.isfinite(R_kpc) and R_kpc > 0):
        raise ValueError(f"Invalid R200m: {R_kpc}")
    return R_kpc

