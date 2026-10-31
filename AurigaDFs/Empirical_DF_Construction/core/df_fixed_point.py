"""Fixed-point (iterated density) isotropic DF, the comparison method of Fig. 2."""
from __future__ import annotations

import numpy as np

from .df_math import default_eps_grid, dMdE_from_monotone_cumfit, g_of_E
from .eddington import eddington_df_from_rgrid_rho
from .spherical import G, potential_from_rho, smooth_rho_profile


def df_fixed_point_from_particles(
    x_kpc: np.ndarray,
    v_kms: np.ndarray,
    m_msun: np.ndarray,
    *,
    r_bins: int = 300,
    r_min_q: float = 1e-6,
    r_max_q: float = 0.999999,
    r_grid_n: int = 6000,
    eps_grid: np.ndarray | None = None,
    nint: int = 5,
    conv_frac: float = 1e-3,
    psi_grid_n: int = 2000,
    psi_r0_factor: float = 1.0,
) -> dict:

    x, v, m = np.asarray(x_kpc, float), np.asarray(v_kms, float), np.asarray(m_msun, float)
    r = np.linalg.norm(x, axis=1)
    v2 = np.sum(v * v, axis=1)

    ok = np.isfinite(r) & (r > 0) & np.isfinite(v2) & (v2 >= 0) & np.isfinite(m) & (m > 0)
    x = x[ok]
    v = v[ok]
    r = r[ok]
    v2 = v2[ok]
    m = m[ok]
    N = int(r.size)
    if N < 5000:
        raise ValueError("Too few particles for fixed-point DF")

    nint = max(1, int(nint))
    psi_grid_n = int(psi_grid_n)
    psi_r0_factor = float(psi_r0_factor)
    conv_frac = float(conv_frac)

    # By convention we always use Phi(infty)=0 (Psi(infty)=0).
    # The *outer boundary* R_edge is where the iterated spherical density becomes zero,
    # which in this discrete setting is the current selection edge (no mass beyond it).
    zero_at_rmax = False

    iter_Psi0: list[float] = []
    iter_n_bound: list[int] = []
    iter_frac_changed: list[float] = []
    iter_rmax_kpc: list[float] = []

    sel = np.ones(N, dtype=bool)

    for _it in range(nint):
        if int(np.count_nonzero(sel)) < 1000:
            break
        rmax = float(np.max(r[sel]))
        if not (np.isfinite(rmax) and rmax > 0):
            break

        r_mid, rho_mid = smooth_rho_profile(r[sel], m[sel], r_max=float(rmax), nbins=int(r_bins), r_min_q=float(r_min_q))
        if r_mid.size < 10:
            raise ValueError("Too few radial shells to build rho(r)")

        r_grid = np.logspace(np.log10(float(r_mid[0])), np.log10(float(rmax)), int(psi_grid_n))
        r0 = max(float(rmax) * 1e-8, float(r_grid[0]) * float(psi_r0_factor))
        if r0 < r_grid[0]:
            r_grid = np.concatenate(([r0], r_grid))
        rho_grid = np.interp(r_grid, r_mid, rho_mid, left=float(rho_mid[0]), right=float(rho_mid[-1]))

        Psi_grid, M_enc = potential_from_rho(r_grid, rho_grid, Gval=G, zero_at_rmax=False)
        Psi0 = float(Psi_grid[0])
        Psi_p = np.interp(r, r_grid, Psi_grid, left=Psi0, right=0.0)
        E_rel = Psi_p - 0.5 * v2
        finite_E = np.isfinite(E_rel)
        bound = finite_E & (E_rel > 0)

        changed = int(np.count_nonzero(sel != bound))
        frac_changed = changed / max(1, N)
        iter_Psi0.append(Psi0)
        iter_n_bound.append(int(np.count_nonzero(bound)))
        iter_frac_changed.append(float(frac_changed))
        iter_rmax_kpc.append(float(rmax))

        sel = bound
        if frac_changed < conv_frac:
            break

    if int(np.count_nonzero(sel)) < 1000:
        raise ValueError("Fixed-point iteration ended with too few bound particles")

    # Final density profile and potential from the converged selection (outer boundary at Rmax_bound).
    rmax_bound = float(np.max(r[sel]))
    r_mid, rho_mid = smooth_rho_profile(r[sel], m[sel], r_max=float(rmax_bound), nbins=int(r_bins), r_min_q=float(r_min_q))
    r_grid = np.logspace(np.log10(float(r_mid[0])), np.log10(float(rmax_bound)), int(psi_grid_n))
    r0 = max(float(rmax_bound) * 1e-8, float(r_grid[0]) * float(psi_r0_factor))
    if r0 < r_grid[0]:
        r_grid = np.concatenate(([r0], r_grid))
    rho_grid = np.interp(r_grid, r_mid, rho_mid, left=float(rho_mid[0]), right=float(rho_mid[-1]))

    Psi_grid, M_enc = potential_from_rho(r_grid, rho_grid, Gval=G, zero_at_rmax=False)
    Psi0 = float(Psi_grid[0])

    Psi_p = np.interp(r, r_grid, Psi_grid, left=Psi0, right=0.0)
    E_rel = Psi_p - 0.5 * v2
    finite_E = np.isfinite(E_rel)
    bound = finite_E & (E_rel > 0)

    # DF uses ALL bound particles (no r<r_delta or similar restriction).
    use_E = bound
    n_df = int(np.count_nonzero(use_E))
    if n_df < 1000:
        raise ValueError("Too few particles for DF after selection")

    if eps_grid is None:
        eps_grid = default_eps_grid()
    eps_grid = np.asarray(eps_grid, float)
    X_eval = eps_grid * Psi0

    # g(E) quadrature grid
    r_grid_g = np.logspace(np.log10(float(r_mid[0])), np.log10(float(rmax_bound)), int(r_grid_n))
    r0g = max(float(rmax_bound) * 1e-8, float(r_grid_g[0]) * float(psi_r0_factor))
    if r0g < r_grid_g[0]:
        r_grid_g = np.concatenate(([r0g], r_grid_g))
    Psi_grid_g = np.interp(r_grid_g, r_grid, Psi_grid, left=Psi0, right=0.0)
    Psi_grid_g = np.maximum(Psi_grid_g, 0.0)
    Psi_grid_g = np.maximum.accumulate(Psi_grid_g[::-1])[::-1]

    X_all = E_rel[use_E]
    m_all = m[use_E]
    # Vacuum tail beyond rmax_bound (potential built with zero_at_rmax=False, so Ψ=GM/r there)
    GM_out = G * float(M_enc[-1]) if np.size(M_enc) else None
    gX = g_of_E(X_eval, r_grid_g, Psi_grid_g, GM_outside=GM_out)
    dMdX = dMdE_from_monotone_cumfit(X_all, m_all, X_eval)

    with np.errstate(divide="ignore", invalid="ignore"):
        fE = dMdX / gX
    okf = np.isfinite(fE) & (fE > 0)
    log10f = np.full_like(eps_grid, np.nan)
    log10f[okf] = np.log10(fE[okf])

    # isotropic reference from Eddington inversion of the same final density profile
    log10f_edd = np.full_like(eps_grid, np.nan)
    fE_edd, _Psi0_edd = eddington_df_from_rgrid_rho(eps_grid, r_grid, rho_grid, Gval=G, zero_at_rmax=False)
    ok_e = np.isfinite(fE_edd) & (fE_edd > 0)
    log10f_edd[ok_e] = np.log10(fE_edd[ok_e])

    # per-epsilon diagnostics (counts and N_eff)
    ok_em = np.isfinite(X_all) & np.isfinite(m_all)
    eps_all = np.asarray(X_all[ok_em] / Psi0, float)
    w_all = np.asarray(m_all[ok_em], float)
    if eps_grid.size >= 2:
        mid = 0.5 * (eps_grid[:-1] + eps_grid[1:])
        edges = np.concatenate(([0.0], mid, [1.0]))
    else:
        edges = np.array([0.0, 1.0], dtype=float)
    eps_all_c = np.clip(eps_all, 0.0, 1.0)
    b = np.searchsorted(edges, eps_all_c, side="right") - 1
    good = (b >= 0) & (b < eps_grid.size)
    b = b[good]
    w_all = w_all[good]
    n_eps = np.bincount(b, minlength=int(eps_grid.size)).astype(int)
    sumw = np.bincount(b, weights=w_all, minlength=int(eps_grid.size)).astype(float)
    sumw2 = np.bincount(b, weights=(w_all**2), minlength=int(eps_grid.size)).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff_eps = np.where(sumw2 > 0, (sumw * sumw) / sumw2, 0.0)
    n_eff_eps = np.where(np.isfinite(n_eff_eps), n_eff_eps, 0.0)

    n_finite = int(np.count_nonzero(finite_E))
    n_bound = int(np.count_nonzero(bound))
    n_unbound = int(n_finite - n_bound)

    return {
        "eps_hat": eps_grid,
        "log10f": log10f,
        "log10f_eddington_from_density": log10f_edd,
        # diagnostics
        "fE": np.asarray(fE, float),
        "gE": np.asarray(gX, float),
        "dMdE": np.asarray(dMdX, float),
        "n_eps": n_eps,
        "n_eff_eps": n_eff_eps,
        "Psi0": float(Psi0),
        "rmax_bound_kpc": float(rmax_bound),
        "M_enc_rmax_bound_msun": float(M_enc[-1]) if np.size(M_enc) else float("nan"),
        "M_bound_msun": float(np.sum(m[bound])),
        "n_total": int(x.shape[0]),
        "n_bound_pot": int(n_bound),
        "n_unbound_pot": int(n_unbound),
        "n_df": int(n_df),
        "nint": int(nint),
        "conv_frac": float(conv_frac),
        "psi_grid_n": int(psi_grid_n),
        "psi_r0_factor": float(psi_r0_factor),
        "phi_zero_at": "infty",
        "g_kepler_tail": GM_out is not None,
        "u_edge": (float(GM_out / (rmax_bound * Psi0)) if GM_out is not None else 0.0),
        "n_iters_used": int(len(iter_Psi0)),
        "iter_Psi0": np.asarray(iter_Psi0, float),
        "iter_n_bound": np.asarray(iter_n_bound, int),
        "iter_frac_changed": np.asarray(iter_frac_changed, float),
        "iter_rmax_kpc": np.asarray(iter_rmax_kpc, float),
    }

