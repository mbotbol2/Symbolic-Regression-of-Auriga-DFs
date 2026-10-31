"""Unit and consistency checks of the bound set against the SUBFIND/FoF catalogue.

Per halo, at z = 0:
  - M(<R_200m) from all DM particles and from the bound set, vs Group_M_Mean200
  - V_c,max = max sqrt(G M(<r) / r) of the bound set, vs SubhaloVmax
  - 1D velocity dispersion of the bound set, vs SubhaloVelDisp
  - virial ratio 2K/|W| of the bound set in the DF's potential (W = -1/2 sum m Psi)
  - mass fraction with eps = Psi - v^2/2 < 0 in that potential, and where it sits
  - Psi(0) of that potential vs the DF metadata

The potential is built exactly as in build_df_raw (bound set only, Phi(oo) = 0), and
velocities are peculiar, relative to SubhaloVel, as in the DF and the Fig. 7 shells.
An h error would show as a ratio of h^{+-1} in the masses or h^{+-1/2} in the speeds.

Needs $AURIGA_DATA. Writes <results>/_checks/catalogue_checks.csv.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import pandas as pd

from ..core.spherical import G, potential_from_rho, smooth_rho_profile
from ..io.auriga_root import resolve_auriga_data_root
from ..io.centering import _read_first_row_from_any_piece
from ..io.files import groupcat_files, snapshot_files
from ..io.header import read_header
from ..io.layout import resolve_data_dir_for_snap, resolve_halo_dir
from ..io.results_paths import halo_results_root, ids_dir
from ..io.subset import load_subset_phase_space_at_snap


def catalogue(gfiles, h, a):
    get = lambda d: float(np.asarray(_read_first_row_from_any_piece(gfiles, d)).ravel()[0])
    return dict(M200m_cat=get("Group/Group_M_Mean200") * 1e10 / h,
                R200m_cat=get("Group/Group_R_Mean200") * 1000.0 / h * a,
                Vmax_cat=get("Subhalo/SubhaloVmax"),
                VelDisp_cat=get("Subhalo/SubhaloVelDisp"))


def sigma1d(v, m):
    """1D dispersion about the mass-weighted mean velocity."""
    dv = v - np.average(v, axis=0, weights=m)
    return float(np.sqrt(np.average(np.einsum("ij,ij->i", dv, dv), weights=m) / 3.0))


def all_dm_mass_within(sfiles, cpos, pos_fac, radius, m_part, chunk=4_000_000):
    n = 0
    for fn in sfiles:
        with h5py.File(fn, "r") as f:
            if "PartType1/Coordinates" not in f:
                continue
            X = f["PartType1/Coordinates"]
            for s in range(0, X.shape[0], chunk):
                dx = (np.asarray(X[s:s + chunk], float) - cpos) * pos_fac
                n += int(np.count_nonzero(np.einsum("ij,ij->i", dx, dx) < radius ** 2))
    return n * m_part


def check_halo(hid: int, snap: int) -> dict:
    hdir = resolve_halo_dir(resolve_auriga_data_root(), hid)
    ddir = resolve_data_dir_for_snap(hdir, snap)
    sfiles, gfiles = snapshot_files(ddir, snap), groupcat_files(ddir, snap)
    hdr = read_header(sfiles[0])
    cat = catalogue(gfiles, hdr.h, hdr.a)
    cpos = np.asarray(_read_first_row_from_any_piece(gfiles, "Subhalo/SubhaloPos"), float)

    ids = np.sort(np.load(ids_dir(hid) / "bound_ids_with_hubble_flow_subhalo.npy").astype(np.uint64))
    d = load_subset_phase_space_at_snap(halo_dir=hdir, subset_ids_sorted=ids, snap=snap, pt=1,
                                        center_mode="subhalo", apply_vel_sqrt_a=True)
    x, v, m = d["x_kpc"], d["v_kms"], d["m_msun"]
    r = np.linalg.norm(x, axis=1)
    v2 = np.einsum("ij,ij->i", v, v)

    # the DF's potential: bound set only, Phi(oo) = 0 (as core.df_from_particles)
    rmax = float(r.max())
    r_mid, rho_mid = smooth_rho_profile(r, m, r_max=rmax, nbins=320, r_min_q=1e-6)
    rg = np.logspace(np.log10(r_mid[0]), np.log10(rmax), 8000)
    psi_g, M_g = potential_from_rho(rg, np.interp(rg, r_mid, rho_mid), Gval=G)
    psi = np.interp(r, rg, psi_g, left=psi_g[0], right=0.0)
    eps = psi - 0.5 * v2

    order = np.argsort(r)
    M_enc = np.cumsum(m[order])
    vc = np.sqrt(G * M_enc / np.maximum(r[order], 1e-3))
    inner = r[order] > 1.0                     # past the softening-dominated centre

    K, W = 0.5 * np.sum(m * v2), -0.5 * np.sum(m * psi)
    neg = eps < 0
    R200 = cat["R200m_cat"]
    m_part = float(np.median(m))
    in200 = r < R200
    meta = json.loads((halo_results_root() / f"Halo_{hid:02d}" / "df" / "raw"
                       / "df_raw_with_hubble_flow_meta.json").read_text())
    return dict(
        halo=hid, h=hdr.h, a=hdr.a,
        R200m_kpc=R200,
        M200m_cat=cat["M200m_cat"],
        M200m_all_dm=all_dm_mass_within(sfiles, cpos, 1000.0 / hdr.h * hdr.a, R200, m_part),
        M200m_bound=float(np.sum(m[in200])),
        Vmax_cat=cat["Vmax_cat"], Vmax_bound=float(vc[inner].max()),
        VelDisp_cat=cat["VelDisp_cat"], sigma1d_bound=sigma1d(v, m),
        sigma1d_bound_in_R200m=sigma1d(v[in200], m[in200]),
        virial_2K_over_W=float(2 * K / abs(W)),
        virial_2K_over_W_in_R200m=float(np.sum(m[in200] * v2[in200]) / abs(0.5 * np.sum(m[in200] * psi[in200]))),
        eps_neg_frac=float(np.sum(m[neg]) / np.sum(m)),
        eps_neg_frac_in_R200m=float(np.sum(m[neg & in200]) / np.sum(m[in200])),
        eps_neg_median_r_over_R200m=float(np.median(r[neg]) / R200) if neg.any() else np.nan,
        Psi0_rebuilt=float(psi_g[0]), Psi0_meta=float(meta["Psi0"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--halos", type=int, nargs="+", default=list(range(1, 31)))
    ap.add_argument("--snap", type=int, default=127)
    args = ap.parse_args()

    rows = []
    for hid in args.halos:
        print(f"[halo {hid}] ...", flush=True)
        rows.append(check_halo(hid, args.snap))
        r = rows[-1]
        print(f"  M200m all/cat {r['M200m_all_dm'] / r['M200m_cat']:.3f}  bound/cat {r['M200m_bound'] / r['M200m_cat']:.3f}"
              f"  Vmax {r['Vmax_bound'] / r['Vmax_cat']:.3f}  sigma {r['sigma1d_bound'] / r['VelDisp_cat']:.3f}"
              f"  2K/|W| {r['virial_2K_over_W']:.3f}  eps<0 {r['eps_neg_frac']:.1%}"
              f" (in R200m {r['eps_neg_frac_in_R200m']:.1%}, median r {r['eps_neg_median_r_over_R200m']:.2f} R200m)",
              flush=True)
    out = halo_results_root() / "_checks"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "catalogue_checks.csv", index=False)
    print(f"-> {out / 'catalogue_checks.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
