"""
Scaling of distribution function to some reference halo, or dimensionless object.
"""
from __future__ import annotations

import numpy as np

from .spherical import G


def energy_scale(rs_kpc: float, rhos_msun_kpc3: float, Gval: float = G) -> float:
    """Structural specific-energy scale E_s = G ρ_s r_s^2, in (km/s)^2."""
    rs = float(rs_kpc)
    rhos = float(rhos_msun_kpc3)
    if not (np.isfinite(rs) and rs > 0 and np.isfinite(rhos) and rhos > 0):
        raise ValueError(f"Invalid scales rs={rs_kpc}, rhos={rhos_msun_kpc3}")
    return float(Gval) * rhos * rs * rs


def df_scale(rs_kpc: float, rhos_msun_kpc3: float, Gval: float = G) -> float:
    """Factor to multiply f by to make it dimensionless: G^{3/2} r_s^3 ρ_s^{1/2}."""
    rs = float(rs_kpc)
    rhos = float(rhos_msun_kpc3)
    if not (np.isfinite(rs) and rs > 0 and np.isfinite(rhos) and rhos > 0):
        raise ValueError(f"Invalid scales rs={rs_kpc}, rhos={rhos_msun_kpc3}")
    return float(Gval) ** 1.5 * rs**3 * np.sqrt(rhos)


def dimensionless_df(
    log10f: np.ndarray,
    *,
    rs_kpc: float,
    rhos_msun_kpc3: float,
    Gval: float = G,
) -> np.ndarray:
    """
    Put a stored log10 f on the dimensionless DF scale, leaving the abscissa alone.

    log10 f̂ = log10 f + log10(G^{3/2} r_s^3 ρ_s^{1/2}).
    The curve stays a function of ε̂, which needs no rescaling. No potential,
    density-of-states or particle quantity is recomputed.
    """
    return np.asarray(log10f, float) + np.log10(df_scale(rs_kpc, rhos_msun_kpc3, Gval=Gval))


def df_to_reference_units(
    eps_hat: np.ndarray,
    log10f: np.ndarray,
    *,
    Psi0: float,
    rs_kpc: float,
    rhos_msun_kpc3: float,
    ref_rs_kpc: float,
    ref_rhos_msun_kpc3: float,
    ref_Psi0: float,
    Gval: float = G,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Express halo i's DF in a reference halo's units:

        f_{i→ref} = f_i (r_{s,i}/r_{s,ref})^3 (ρ_{s,i}/ρ_{s,ref})^{1/2}

    Returns (ε̂_ref, log10 f_{i→ref}). The similarity map rescales energies by
    E_s,ref/E_s, so the abscissa on the reference halo's normalisation is

        ε̂_ref = ε̂ (Ψ_i(0)/E_s) (E_s,ref/Ψ_ref(0)),

    which can be overlaid directly on the reference halo's stored ε̂ grid.
    """
    Psi0 = float(Psi0)
    if not (np.isfinite(Psi0) and Psi0 > 0):
        raise ValueError(f"Invalid Psi0={Psi0}")
    ref_Psi0 = float(ref_Psi0)
    if not (np.isfinite(ref_Psi0) and ref_Psi0 > 0):
        raise ValueError(f"Invalid ref_Psi0={ref_Psi0}")
    log10f_hat = dimensionless_df(
        log10f, rs_kpc=rs_kpc, rhos_msun_kpc3=rhos_msun_kpc3, Gval=Gval
    )
    log10f_ref = log10f_hat - np.log10(df_scale(ref_rs_kpc, ref_rhos_msun_kpc3, Gval=Gval))
    E_s = energy_scale(rs_kpc, rhos_msun_kpc3, Gval=Gval)
    E_s_ref = energy_scale(ref_rs_kpc, ref_rhos_msun_kpc3, Gval=Gval)
    eps_hat_ref = np.asarray(eps_hat, float) * (Psi0 / E_s) * (E_s_ref / ref_Psi0)
    return eps_hat_ref, log10f_ref
