from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline

from .spherical import G, potential_from_rho

_U, _W = np.polynomial.legendre.leggauss(256)


def eddington_df_from_rgrid_rho(
    eps_hat: np.ndarray,
    r_grid: np.ndarray,
    rho_grid: np.ndarray,
    Gval: float = G,
    *,
    zero_at_rmax: bool = False,
    Psi_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Isotropic DF f(E) at E = eps_hat * Psi(0) for a spherical rho(r), by Eddington's formula:

        f(E) = [ int_0^E rho''(Psi) dPsi / sqrt(E - Psi) + rho'(0) / sqrt(E) ] / (sqrt(8) pi^2),

    with rho(Psi) a cubic spline. The integral is taken with Psi = E - u^2, which removes the
    1/sqrt(E - Psi) singularity, on Gauss-Legendre nodes in u; this matches the analytic
    Hernquist DF to ~2e-4 dex. Psi(r) is solved from rho unless Psi_grid is given.
    Returns (f, Psi0).
    """
    eps_hat = np.asarray(eps_hat, float)
    rho = np.asarray(rho_grid, float)
    if Psi_grid is None:
        Psi_grid, _ = potential_from_rho(np.asarray(r_grid, float), rho, Gval=Gval, zero_at_rmax=bool(zero_at_rmax))
    Psi_grid = np.asarray(Psi_grid, float)
    Psi0 = float(Psi_grid[0])

    P, idx = np.unique(np.maximum.accumulate(np.maximum(Psi_grid[::-1], 0.0)), return_index=True)
    f = np.full_like(eps_hat, np.nan)
    if P.size < 20:
        return f, Psi0
    spl = UnivariateSpline(P, rho[::-1][idx], s=0, k=3)

    E = eps_hat * Psi0
    ok = np.isfinite(E) & (E > 0)
    sE = np.sqrt(E[ok])
    u = 0.5 * (_U + 1)[None, :] * sE[:, None]
    integral = (spl.derivative(2)(E[ok][:, None] - u ** 2) * _W[None, :]).sum(axis=1) * sE
    f[ok] = (integral + spl.derivative(1)(P[0]) / sE) / (np.sqrt(8) * np.pi ** 2)
    return f, Psi0
