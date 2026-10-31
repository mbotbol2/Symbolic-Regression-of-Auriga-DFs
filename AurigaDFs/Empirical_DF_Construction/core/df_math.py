from __future__ import annotations

import numpy as np


def default_eps_grid(
    *,
    eps_lo: float = 1e-3,
    eps_mid_lo: float = 5e-2,
    eps_mid_hi: float = 0.95,
    eps_hi: float = 0.999,
    n_lo: int = 200,
    n_mid: int = 200,
    n_hi: int = 200,
) -> np.ndarray:
    """
    Default energy grid for DF work:
    - log-spaced at low eps (outer tail near escape),
    - linear in the bulk,
    - log-spaced in (1-eps) near eps->1 (deeply bound end).
    """
    eps_lo = float(eps_lo)
    eps_mid_lo = float(eps_mid_lo)
    eps_mid_hi = float(eps_mid_hi)
    eps_hi = float(eps_hi)
    # The three segments are concatenated assuming eps_lo <= eps_mid_lo <= eps_mid_hi <= eps_hi.
    # Without this check a floor above eps_mid_lo silently reintroduces points below it via the
    # linear middle segment.
    if not (0.0 < eps_lo <= eps_mid_lo < eps_mid_hi <= eps_hi < 1.0):
        raise ValueError(
            f"Need 0 < eps_lo <= eps_mid_lo < eps_mid_hi <= eps_hi < 1, got "
            f"eps_lo={eps_lo}, eps_mid_lo={eps_mid_lo}, eps_mid_hi={eps_mid_hi}, eps_hi={eps_hi}"
        )

    lo = np.logspace(np.log10(eps_lo), np.log10(eps_mid_lo), int(n_lo), dtype=float)
    mid = np.linspace(eps_mid_lo, eps_mid_hi, int(n_mid), dtype=float)
    one_minus_hi = np.logspace(np.log10(1.0 - eps_hi), np.log10(1.0 - eps_mid_hi), int(n_hi), dtype=float)
    hi = 1.0 - one_minus_hi
    # np.logspace round-trips 10**log10(x), so pin the endpoints: a floor that lands at
    # 0.029999999999999995 would be dropped by a downstream `eps >= 0.03` mask.
    lo[0] = eps_lo
    hi[-1] = eps_hi

    eps = np.unique(np.concatenate([lo, mid, hi]).astype(float))
    eps = eps[(eps > 0.0) & (eps < 1.0)]
    eps.sort()
    return eps


def g_kepler_tail(Evals: np.ndarray, GM_outside: float, r_max: float) -> np.ndarray:
    """
    Vacuum contribution to g(E) from r > r_max, where Ψ(r) = GM_outside / r.

    This is the exact value of

        16π^2 ∫_{r_max}^{GM/E} r^2 sqrt(2(GM/r - E)) dr
          = 32√2 π^2 (GM)^3 E^{-5/2} [F(π/2) - F(θ_R)],
          F(θ) = θ/16 - sin(2θ)/64 - sin(4θ)/64 + sin(6θ)/192,
          sin^2(θ_R) = E r_max / GM,

    obtained with r = (GM/E) sin^2 θ. It is the phase-space volume the states of energy
    E occupy outside r_max under the model ρ(r>r_max)=0, Φ(∞)=0 (i.e. the potential built
    by potential_from_rho with zero_at_rmax=False). Zero for E >= GM/r_max, where the
    turning point already lies inside r_max.
    """
    E = np.asarray(Evals, float)
    GM = float(GM_outside)
    r_max = float(r_max)
    if not (np.isfinite(GM) and GM > 0 and np.isfinite(r_max) and r_max > 0):
        raise ValueError(f"Invalid GM_outside={GM_outside} or r_max={r_max}")

    E_edge = GM / r_max  # = Ψ(r_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = np.clip(E * r_max / GM, 0.0, 1.0)  # sin^2(θ_R)
        theta = np.arcsin(np.sqrt(s2))
        F = lambda t: t / 16.0 - np.sin(2.0 * t) / 64.0 - np.sin(4.0 * t) / 64.0 + np.sin(6.0 * t) / 192.0
        tail = 32.0 * np.sqrt(2.0) * np.pi**2 * GM**3 * (F(0.5 * np.pi) - F(theta)) / np.power(E, 2.5)

    active = np.isfinite(E) & (E > 0.0) & (E < E_edge)
    return np.where(active, np.where(np.isfinite(tail), tail, 0.0), 0.0)


def g_of_E(
    Evals: np.ndarray,
    r_grid: np.ndarray,
    Psi_grid: np.ndarray,
    *,
    GM_outside: float | None = None,
) -> np.ndarray:
    """
    Density of states g(E) = 16π^2 ∫ r^2 sqrt(2(Ψ(r)-E)) dr over Ψ>E.

    The quadrature is limited to r_grid, i.e. to Rmax = r_grid[-1]. For E < Ψ(Rmax) the
    turning point lies outside the grid and the integral is clipped, which underestimates
    g and inflates f = (dM/dE)/g at low ε̂.

    Pass GM_outside = G * M_enc(Rmax) to add the analytic vacuum ("Kepler") tail from
    r > Rmax, which is what the potential model assumes when Φ(∞)=0 and ρ(r>Rmax)=0.
    Use the M_enc that produced Psi_grid, so that Ψ and its tail stay consistent. Leave
    it None when the potential was built with Ψ(Rmax)=0 (zero_at_rmax=True): no vacuum
    region then has Ψ>E.
    """
    Evals = np.asarray(Evals, float)
    g = np.full_like(Evals, np.nan)
    r_grid = np.asarray(r_grid, float)
    Psi_grid = np.asarray(Psi_grid, float)
    r2 = r_grid * r_grid
    for i, E in enumerate(Evals):
        m = Psi_grid > E
        if np.count_nonzero(m) < 10:
            continue
        rr = r_grid[m]
        pp = Psi_grid[m]
        integrand = r2[m] * np.sqrt(np.maximum(2.0 * (pp - E), 0.0))
        g[i] = 16.0 * np.pi**2 * np.trapezoid(integrand, rr)

    if GM_outside is not None:
        g = g + g_kepler_tail(Evals, float(GM_outside), float(r_grid[-1]))
    return g


def dMdE_from_monotone_cumfit(
    E_rel: np.ndarray,
    m: np.ndarray,
    E_eval: np.ndarray,
    n_knots: int = 4000,
) -> np.ndarray:
    """Compute dM/dE at E_eval from numerical derivative of empirical M(>E)."""
    E_rel = np.asarray(E_rel, float)
    m = np.asarray(m, float)
    ok = np.isfinite(E_rel) & np.isfinite(m) & (m > 0)
    E_rel = E_rel[ok]
    m = m[ok]

    o = np.argsort(E_rel)
    E = E_rel[o]
    w = m[o]

    Mgt = np.cumsum(w[::-1])[::-1]

    N = E.size
    nk = int(min(n_knots, max(200, N // 50)))
    idx = np.unique(np.linspace(0, N - 1, nk).astype(int))
    E_k = E[idx]
    M_k = Mgt[idx]

    uniq = np.unique(E_k)
    if uniq.size < 20:
        raise ValueError("Too few unique energies for numerical derivative")
    last: dict[float, float] = {}
    for ee, mm in zip(E_k, M_k):
        last[float(ee)] = float(mm)
    E_u = uniq
    M_u = np.array([last[float(ee)] for ee in E_u])

    dMgt_dE = np.gradient(M_u, E_u, edge_order=1)
    dMdE_u = np.maximum(-dMgt_dE, 0.0)

    E_eval = np.asarray(E_eval, float)
    dMdE = np.interp(E_eval, E_u, dMdE_u, left=0.0, right=0.0)
    dMdE = np.where(np.isfinite(dMdE), dMdE, 0.0)
    return dMdE

