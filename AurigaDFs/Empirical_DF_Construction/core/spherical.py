from __future__ import annotations

import numpy as np

# Gravitational constant in kpc (km/s)^2 / Msun
G = 4.30091e-6


def smooth_rho_profile(
    r: np.ndarray,
    m: np.ndarray,
    r_max: float,
    nbins: int = 300,
    r_min_q: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute rho(r) for r in (0, r_max] using log-binned shells."""
    r = np.asarray(r, float)
    m = np.asarray(m, float)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(m) & (m > 0) & (r <= float(r_max))
    r, m = r[ok], m[ok]

    r_lo = float(np.quantile(r, float(r_min_q)))
    r_hi = float(r_max)
    edges = np.logspace(np.log10(r_lo), np.log10(r_hi), int(nbins) + 1)

    b = np.digitize(r, edges) - 1
    b = np.clip(b, 0, len(edges) - 2)

    shell_mass = np.bincount(b, weights=m, minlength=len(edges) - 1)
    counts = np.bincount(b, minlength=len(edges) - 1)

    r_in = edges[:-1]
    r_out = edges[1:]
    r_mid = np.sqrt(r_in * r_out)
    vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)

    good = (counts > 0) & (shell_mass > 0) & np.isfinite(r_mid) & np.isfinite(vol) & (vol > 0)
    r_mid = r_mid[good]
    rho = shell_mass[good] / vol[good]
    return r_mid, rho


def potential_from_rho(
    r_grid: np.ndarray,
    rho_grid: np.ndarray,
    Gval: float = G,
    *,
    zero_at_rmax: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Psi(r)=-Phi(r) from a smooth rho(r) on r_grid.

    - If zero_at_rmax=False (default): assumes rho(r>Rmax)=0 and sets Phi(infty)=0,
      i.e. Psi(Rmax)=G*Mtot/Rmax.
    - If zero_at_rmax=True: sets Phi(Rmax)=0, i.e. Psi(Rmax)=0.
    """
    r_grid = np.asarray(r_grid, float)
    rho_grid = np.asarray(rho_grid, float)

    integrand = 4.0 * np.pi * rho_grid * r_grid * r_grid
    M_enc = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_grid))
    M_enc = np.concatenate(([0.0], M_enc))

    Rmax = float(r_grid[-1])
    Mtot = float(M_enc[-1])

    f = M_enc / np.maximum(r_grid * r_grid, 1e-30)
    I = np.zeros_like(r_grid)
    I[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r_grid))
    J = I[-1] - I

    Psi = float(Gval) * (J if bool(zero_at_rmax) else (J + Mtot / Rmax))
    Psi = np.maximum(Psi, 0.0)
    Psi = np.maximum.accumulate(Psi[::-1])[::-1]
    return Psi, M_enc
