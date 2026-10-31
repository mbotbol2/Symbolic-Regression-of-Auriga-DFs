from __future__ import annotations

import numpy as np


def density_profile_from_positions(
    *,
    x_kpc: np.ndarray,
    m_msun: np.ndarray,
    nbins: int = 80,
    r_min_q: float = 1e-6,
    r_max_q: float = 0.999999,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple spherical density profile from particle positions:
      returns (r_mid_kpc, rho_msun_kpc3, counts)
    """
    x = np.asarray(x_kpc, float)
    m = np.asarray(m_msun, float)
    r = np.linalg.norm(x, axis=1)
    ok = np.isfinite(r) & (r > 0) & np.isfinite(m) & (m > 0)
    r = r[ok]
    m = m[ok]

    r_lo = float(np.quantile(r, float(r_min_q)))
    r_hi = float(np.quantile(r, float(r_max_q)))
    edges = np.logspace(np.log10(r_lo), np.log10(r_hi), int(nbins) + 1)

    b = np.digitize(r, edges) - 1
    b = np.clip(b, 0, len(edges) - 2)
    counts = np.bincount(b, minlength=len(edges) - 1)
    shell_mass = np.bincount(b, weights=m, minlength=len(edges) - 1)

    r_in = edges[:-1]
    r_out = edges[1:]
    r_mid = np.sqrt(r_in * r_out)
    vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
    rho = shell_mass / np.maximum(vol, 1e-300)

    good = (counts > 0) & np.isfinite(r_mid) & np.isfinite(rho) & (rho > 0)
    return r_mid[good], rho[good], counts[good].astype(int)

