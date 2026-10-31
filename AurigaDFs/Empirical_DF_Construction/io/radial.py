"""Load particles within a radial cut (e.g. R200m) at a snapshot."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import h5py

from .centering import load_center_from_catalog
from .files import groupcat_files, snapshot_files
from .header import read_header
from .layout import resolve_data_dir_for_snap
from .virial import load_R200m_kpc


def _mass_unit_msun_over_h(snapshot_file: Path) -> float | None:
    MSUN_IN_G = 1.98847e33
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


def load_particles_within_R200m(
    *,
    halo_dir: Path,
    snap: int,
    r_max_kpc: float | None = None,
    pt: int = 1,
    center_mode: str = "subhalo",
    apply_vel_sqrt_a: bool = True,
    chunk: int = 2_000_000,
) -> dict[str, np.ndarray]:
    """
    Load x_kpc, v_kms, m_msun for all DM particles with r < r_max in subhalo frame.

    Parameters
    ----------
    r_max_kpc : float | None
        Maximum radius in physical kpc. If None, uses R200m from group catalog.
    """
    halo_dir = Path(halo_dir).expanduser().resolve()
    data_dir = resolve_data_dir_for_snap(halo_dir, int(snap))
    sfiles = snapshot_files(data_dir, int(snap))
    gfiles = groupcat_files(data_dir, int(snap))
    if not sfiles or not gfiles:
        raise FileNotFoundError(f"Missing snapshot/groups for halo={halo_dir} snap={snap}")

    hdr = read_header(sfiles[0])
    cpos_raw, cvel_kms, _ = load_center_from_catalog(
        groupcat_files_list=gfiles, center_mode=str(center_mode), a=float(hdr.a)
    )
    r_max = r_max_kpc if r_max_kpc is not None else load_R200m_kpc(gfiles=gfiles, a=float(hdr.a), h=float(hdr.h))
    if not (np.isfinite(hdr.h) and hdr.h > 0 and np.isfinite(hdr.a)):
        raise ValueError(f"Invalid header for snap={snap}: a={hdr.a} h={hdr.h}")

    pos_fac = (1000.0 / float(hdr.h)) * float(hdr.a)
    mu = _mass_unit_msun_over_h(sfiles[0])
    if mu is None:
        mu = 1e10
    if hdr.mass_table is None:
        raise RuntimeError("MassTable missing; expected constant DM mass in header")
    m_code = float(hdr.mass_table[int(pt)])
    m_part_msun = (m_code * float(mu)) / float(hdr.h)

    d_x = f"PartType{int(pt)}/Coordinates"
    d_v = f"PartType{int(pt)}/Velocities"

    x_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    m_list: list[np.ndarray] = []

    for fn in sfiles:
        with h5py.File(fn, "r") as f:
            coords = f[d_x]
            vels = f[d_v]
            n = int(coords.shape[0])
            for start in range(0, n, int(chunk)):
                stop = min(start + int(chunk), n)
                X = np.asarray(coords[start:stop], dtype=float)
                V = np.asarray(vels[start:stop], dtype=float)
                dx = (X - cpos_raw[None, :]) * pos_fac
                r = np.sqrt(np.sum(dx * dx, axis=1))
                sel = r < r_max
                if not np.any(sel):
                    continue
                vv = V[sel]
                if bool(apply_vel_sqrt_a):
                    vv = vv * float(np.sqrt(float(hdr.a)))
                vv = vv - cvel_kms[None, :]
                x_list.append(dx[sel].astype(float))
                v_list.append(vv.astype(float))
                m_list.append(np.full(int(np.count_nonzero(sel)), float(m_part_msun), dtype=float))

    if not x_list:
        raise RuntimeError("No particles found within r_max")

    R200m = load_R200m_kpc(gfiles=gfiles, a=float(hdr.a), h=float(hdr.h))
    return {
        "x_kpc": np.concatenate(x_list, axis=0),
        "v_kms": np.concatenate(v_list, axis=0),
        "m_msun": np.concatenate(m_list, axis=0),
        "h": np.array(float(hdr.h)),
        "z": np.array(float(hdr.z)),
        "Omega0": np.array(float(hdr.omega_m)),
        "OmegaL": np.array(float(hdr.omega_l)),
        "a": np.array(float(hdr.a)),
        "R200m_kpc": np.array(float(R200m)),
    }
