from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import h5py

from ..io.centering import load_center_from_catalog
from ..io.files import groupcat_files, snapshot_files
from ..io.header import read_header
from ..io.id_cache import ensure_sorted_unique_uint64
from ..io.layout import resolve_data_dir_for_snap


@dataclass(frozen=True)
class BoundSelectionConfig:
    snap_min: int = 77
    snap_max: int = 127
    pt: int = 1
    center_mode: str = "subhalo"  # 'subhalo' or 'group'
    apply_vel_sqrt_a: bool = True
    chunk: int = 2_000_000
    eps_vr: float = 0.0  # deadzone for sign detection (km/s)


def compute_bound_ids_by_signflip(
    *,
    halo_dir: Path,
    ids0_sorted: np.ndarray,
    cfg: BoundSelectionConfig,
    hubble_flow: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic peri/apo selection:
      detect first pericenter (v_r: - -> +), then first apocenter (+ -> -)

    The sign is taken of the peculiar radial velocity, or with hubble_flow of the proper one,
    v_r,pec + H(z) r.

    Returns:
      - bound_mask over ids0_sorted
      - r_apo_first_kpc (physical kpc) for first apocenter (NaN if none)
    """
    halo_dir = Path(halo_dir).expanduser().resolve()
    ids0_sorted = ensure_sorted_unique_uint64(ids0_sorted)
    N = int(ids0_sorted.size)
    if N == 0:
        raise ValueError("ids0_sorted is empty")

    pt = int(cfg.pt)
    d_id = f"PartType{pt}/ParticleIDs"
    d_x = f"PartType{pt}/Coordinates"
    d_v = f"PartType{pt}/Velocities"

    last_sign = np.zeros(N, dtype=np.int8)
    saw_peri = np.zeros(N, dtype=bool)
    saw_apo = np.zeros(N, dtype=bool)
    r_apo_first = np.full(N, np.nan, dtype=np.float32)

    snaps = list(range(int(cfg.snap_min), int(cfg.snap_max) + 1))

    for snap in snaps:
        data_dir = resolve_data_dir_for_snap(halo_dir, snap)
        sfiles = snapshot_files(data_dir, snap)
        gfiles = groupcat_files(data_dir, snap)
        if not sfiles or not gfiles:
            continue

        hdr = read_header(sfiles[0])
        if not (np.isfinite(hdr.h) and hdr.h > 0 and np.isfinite(hdr.a) and hdr.a > 0):
            raise ValueError(f"Invalid header for snap={snap}: a={hdr.a} h={hdr.h}")

        cpos_raw, cvel_kms, _ = load_center_from_catalog(
            groupcat_files_list=gfiles, center_mode=str(cfg.center_mode), a=float(hdr.a)
        )

        pos_fac = (1000.0 / float(hdr.h)) * float(hdr.a)  # physical kpc

        # H(z) in km/s/kpc, for the proper radial velocity
        H_kms_kpc = 0.0
        if hubble_flow and np.isfinite(hdr.omega_m) and np.isfinite(hdr.omega_l):
            # lightweight inline: H(z)=0.1*h*sqrt(Omega0*(1+z)^3+OmegaL)
            H_kms_kpc = 0.1 * float(hdr.h) * float(np.sqrt(float(hdr.omega_m) * (1.0 + float(hdr.z)) ** 3 + float(hdr.omega_l)))

        for fn in sfiles:
            with h5py.File(fn, "r") as f:
                if (d_id not in f) or (d_x not in f) or (d_v not in f):
                    continue

                ids = f[d_id]
                coords = f[d_x]
                vels = f[d_v]
                n = int(ids.shape[0])

                for start in range(0, n, int(cfg.chunk)):
                    stop = min(start + int(cfg.chunk), n)
                    pid = np.asarray(ids[start:stop], dtype=np.uint64)
                    j = np.searchsorted(ids0_sorted, pid)
                    in_range = j < ids0_sorted.size
                    ok = np.zeros_like(in_range, dtype=bool)
                    if np.any(in_range):
                        jj = j[in_range]
                        ok[in_range] = ids0_sorted[jj] == pid[in_range]
                    if not np.any(ok):
                        continue

                    idx = j[ok]
                    X = np.asarray(coords[start:stop][ok], dtype=float)
                    V = np.asarray(vels[start:stop][ok], dtype=float)

                    dx = (X - cpos_raw[None, :]) * pos_fac  # kpc physical
                    r = np.sqrt(np.sum(dx * dx, axis=1)).astype(float)

                    vv = V
                    if bool(cfg.apply_vel_sqrt_a):
                        vv = vv * float(np.sqrt(float(hdr.a)))  # -> km/s
                    vv = vv - cvel_kms[None, :]

                    rr_safe = np.where(r > 0, r, np.nan)
                    vr = np.sum(dx * vv, axis=1) / rr_safe + float(H_kms_kpc) * r

                    eps = float(cfg.eps_vr)
                    sgn = np.zeros_like(vr, dtype=np.int8)
                    sgn[vr > eps] = 1
                    sgn[vr < -eps] = -1

                    nonzero = sgn != 0
                    if not np.any(nonzero):
                        continue

                    ii = idx[nonzero]
                    cur = sgn[nonzero]
                    last = last_sign[ii]

                    peri = (~saw_peri[ii]) & (last == -1) & (cur == 1)
                    if np.any(peri):
                        saw_peri[ii[peri]] = True

                    apo = (saw_peri[ii]) & (~saw_apo[ii]) & (last == 1) & (cur == -1)
                    if np.any(apo):
                        jj = ii[apo]
                        saw_apo[jj] = True
                        r_apo_first[jj] = r[nonzero][apo].astype(np.float32)

                    last_sign[ii] = cur

    bound = saw_apo
    return bound, r_apo_first

