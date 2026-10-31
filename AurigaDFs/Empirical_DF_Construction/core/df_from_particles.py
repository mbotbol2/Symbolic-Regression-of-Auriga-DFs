from __future__ import annotations

import numpy as np

from .df_math import g_of_E, dMdE_from_monotone_cumfit
from .spherical import G, potential_from_rho, smooth_rho_profile


def df_isotropic_from_particles_rmax(
    *,
    x_kpc: np.ndarray,
    v_kms: np.ndarray,
    m_msun: np.ndarray,
    eps_hat: np.ndarray,
    phi_zero_at: str = "infty",
    nbins_rho: int = 320,
    r_min_q: float = 1e-6,
    r_max_q: float | None = None,
    r_grid_n: int = 8000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute isotropic DF f(E) from a particle set, using a spherical potential constructed
    from the same set on [r_min, Rmax].

    - Outer boundary Rmax:
        - r_max_q is None (default): furthest particle radius (nanmax)
        - otherwise: quantile of the particle-radius distribution, e.g. 0.99
    - Inner edge of rho shells: r_min_q quantile (default 1e-6)
    - Potential zero-point:
        - phi_zero_at='infty' (default): assume rho(r>Rmax)=0 and set Phi(infty)=0
        - phi_zero_at='rmax': set Phi(Rmax)=0

    Returns raw log10 f(E) (NOT mass-normalised).
    """
    x = np.asarray(x_kpc, float)
    v = np.asarray(v_kms, float)
    m = np.asarray(m_msun, float)
    eps_hat = np.asarray(eps_hat, float)

    r = np.linalg.norm(x, axis=1)
    v2 = np.sum(v * v, axis=1)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(v2) & (v2 >= 0) & np.isfinite(m) & (m > 0)
    if np.count_nonzero(ok) < 50_000:
        raise RuntimeError(f"Too few particles for DF after cuts: {np.count_nonzero(ok)}")

    rr_all = r[ok]
    if r_max_q is None:
        rmax = float(np.nanmax(rr_all))
        r_max_q_used = None
    else:
        q = float(r_max_q)
        if not (0.0 < q <= 1.0):
            raise ValueError(f"r_max_q must be in (0,1], got {q}")
        rmax = float(np.quantile(rr_all, q))
        r_max_q_used = q
    if not (np.isfinite(rmax) and rmax > 0):
        raise RuntimeError("Invalid rmax for DF")

    rr = r[ok]
    mm = m[ok]

    r_mid, rho_mid = smooth_rho_profile(rr, mm, r_max=float(rmax), nbins=int(nbins_rho), r_min_q=float(r_min_q))
    if r_mid.size < 50:
        raise RuntimeError("Failed to build smooth rho profile")

    r_grid = np.logspace(np.log10(float(r_mid[0])), np.log10(float(rmax)), int(r_grid_n), dtype=float)
    rho_grid = np.interp(r_grid, r_mid, rho_mid, left=float(rho_mid[0]), right=float(rho_mid[-1]))

    zmode = str(phi_zero_at).strip().lower()
    if zmode not in {"infty", "rmax"}:
        raise ValueError("phi_zero_at must be one of {'infty','rmax'}")
    Psi_grid, M_enc = potential_from_rho(r_grid, rho_grid, Gval=G, zero_at_rmax=(zmode == "rmax"))
    Psi0 = float(Psi_grid[0])
    if not (np.isfinite(Psi0) and Psi0 > 0):
        raise RuntimeError(f"Invalid Psi0={Psi0}")

    Psi_p = np.interp(rr, r_grid, Psi_grid, left=Psi0, right=0.0)
    E_rel = Psi_p - 0.5 * np.asarray(v2[ok], float)

    use = np.isfinite(E_rel) & (E_rel > 0) & np.isfinite(mm) & (mm > 0)
    if np.count_nonzero(use) < 10_000:
        raise RuntimeError(f"Too few (E_rel>0) particles for dM/dE: {np.count_nonzero(use)}")

    E_eval = eps_hat * Psi0
    # Outside Rmax the model is vacuum with Ψ=G*M_enc(Rmax)/r, so states with E<Ψ(Rmax)
    # extend past the grid; add that piece analytically (no tail when Ψ(Rmax)=0 by fiat).
    GM_out = G * float(M_enc[-1]) if (zmode == "infty" and np.size(M_enc)) else None
    gE = g_of_E(E_eval, r_grid, Psi_grid, GM_outside=GM_out)
    dMdE = dMdE_from_monotone_cumfit(E_rel[use], mm[use], E_eval)
    with np.errstate(divide="ignore", invalid="ignore"):
        fE = dMdE / gE

    log10f = np.full_like(eps_hat, np.nan, dtype=float)
    okf = np.isfinite(fE) & (fE > 0)
    log10f[okf] = np.log10(fE[okf])

    meta = {
        "Psi0": float(Psi0),
        "rmax_kpc": float(rmax),
        "r_min_q": float(r_min_q),
        "r_max_q": r_max_q_used,
        "Mtot_enc_msun": float(M_enc[-1]) if np.size(M_enc) else float("nan"),
        "n_use": int(np.count_nonzero(use)),
        "df_potential_outer_boundary": "rmax",
        "df_potential_zero_at": str(zmode),
        "g_kepler_tail": GM_out is not None,
        "u_edge": (float(GM_out / (rmax * Psi0)) if GM_out is not None else 0.0),
        "mass_normalised": False,
    }
    return eps_hat, log10f, meta

