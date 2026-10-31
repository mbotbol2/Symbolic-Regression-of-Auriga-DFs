from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

from .centering import load_center_from_catalog
from .files import groupcat_files, snapshot_files
from .header import read_header
from .id_cache import ensure_sorted_unique_uint64
from .layout import resolve_data_dir_for_snap


def _mass_unit_msun_over_h(snapshot_file: Path) -> float | None:
    """
    Try to read UnitMass_in_g from snapshot header/parameters and convert to Msun/h.
    Returns None if missing.
    """
    MSUN_IN_G = 1.98847e33
    snapshot_file = Path(snapshot_file)
    try:
        with h5py.File(snapshot_file, "r") as f:
            hdr = f["Header"].attrs if "Header" in f else {}
            um = hdr.get("UnitMass_in_g", None)
            if um is None and "Parameters" in f and "UnitMass_in_g" in f["Parameters"].attrs:
                um = f["Parameters"].attrs["UnitMass_in_g"]
            if um is None:
                return None
            return float(um) / MSUN_IN_G
    except OSError:
        return None


def load_subset_phase_space_at_snap(
    *,
    halo_dir: Path,
    subset_ids_sorted: np.ndarray,
    snap: int,
    pt: int,
    center_mode: str,
    apply_vel_sqrt_a: bool,
    chunk: int = 2_000_000,
) -> dict[str, np.ndarray]:
    """
    Load x_kpc, v_kms, m_msun for only subset_ids at a snapshot, centered on chosen halo center.
    """
    halo_dir = Path(halo_dir).expanduser().resolve()
    subset_ids_sorted = ensure_sorted_unique_uint64(subset_ids_sorted)
    if subset_ids_sorted.size == 0:
        raise ValueError("subset_ids_sorted is empty")

    data_dir = resolve_data_dir_for_snap(halo_dir, int(snap))
    sfiles = snapshot_files(data_dir, int(snap))
    gfiles = groupcat_files(data_dir, int(snap))
    if not sfiles or not gfiles:
        raise FileNotFoundError(f"Missing snapshot/groups for halo={halo_dir} snap={snap}")

    hdr = read_header(sfiles[0])
    cpos_raw, cvel_kms, _ = load_center_from_catalog(
        groupcat_files_list=gfiles, center_mode=str(center_mode), a=float(hdr.a)
    )
    if not (np.isfinite(hdr.h) and hdr.h > 0 and np.isfinite(hdr.a)):
        raise ValueError(f"Invalid header for snap={snap}: a={hdr.a} h={hdr.h}")

    pos_fac = (1000.0 / float(hdr.h)) * float(hdr.a)  # physical kpc

    # mass conversion
    mu = _mass_unit_msun_over_h(sfiles[0])
    if mu is None:
        mu = 1e10  # common Auriga convention (Msun/h per code unit)
    if hdr.mass_table is None:
        raise RuntimeError("MassTable missing; expected constant DM mass in header")
    m_code = float(hdr.mass_table[int(pt)])
    m_part_msun = (m_code * float(mu)) / float(hdr.h)

    d_id = f"PartType{int(pt)}/ParticleIDs"
    d_x = f"PartType{int(pt)}/Coordinates"
    d_v = f"PartType{int(pt)}/Velocities"

    x_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []
    id_list: list[np.ndarray] = []

    for fn in sfiles:
        with h5py.File(fn, "r") as f:
            ids_arr = f[d_id]
            coords = f[d_x]
            vels = f[d_v]
            n = int(ids_arr.shape[0])
            for start in range(0, n, int(chunk)):
                stop = min(start + int(chunk), n)
                pid = np.asarray(ids_arr[start:stop], dtype=np.uint64)
                j = np.searchsorted(subset_ids_sorted, pid)
                in_range = j < subset_ids_sorted.size
                msel = np.zeros_like(in_range, dtype=bool)
                if np.any(in_range):
                    jj = j[in_range]
                    msel[in_range] = subset_ids_sorted[jj] == pid[in_range]
                if not np.any(msel):
                    continue

                pid_sel = pid[msel]
                X = np.asarray(coords[start:stop][msel], dtype=float)
                V = np.asarray(vels[start:stop][msel], dtype=float)

                dx = (X - cpos_raw[None, :]) * pos_fac  # kpc
                vv = V
                if bool(apply_vel_sqrt_a):
                    vv = vv * float(np.sqrt(float(hdr.a)))  # km/s
                vv = vv - cvel_kms[None, :]

                x_list.append(dx.astype(float))
                v_list.append(vv.astype(float))
                m_list.append(np.full(dx.shape[0], float(m_part_msun), dtype=float))
                id_list.append(pid_sel.astype(np.uint64))

    if not x_list:
        raise RuntimeError("No subset particles found when loading phase space")

    out: dict[str, np.ndarray] = {
        "x_kpc": np.concatenate(x_list, axis=0),
        "v_kms": np.concatenate(v_list, axis=0),
        "m_msun": np.concatenate(m_list, axis=0),
        "ids": np.concatenate(id_list, axis=0),
        "h": np.array(float(hdr.h)),
        "z": np.array(float(hdr.z)),
        "Omega0": np.array(float(hdr.omega_m)),
        "OmegaL": np.array(float(hdr.omega_l)),
        "a": np.array(float(hdr.a)),
    }
    return out

