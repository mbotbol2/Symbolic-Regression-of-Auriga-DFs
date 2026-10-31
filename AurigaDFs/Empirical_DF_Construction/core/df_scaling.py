"""
Structural (dimensionless) scaling of an isotropic DF.

Rather than transforming particles and rebuilding ρ, Ψ and g(E) per halo, an already
constructed f(E) can be put in dimensionless form analytically. With a length scale r_s
and a density scale ρ_s (e.g. the NFW fit parameters), the natural units are

    energy (per unit mass):  E_s = G ρ_s r_s^2
    velocity:                V_s = sqrt(E_s) = r_s sqrt(G ρ_s)
    mass:                    M_s = ρ_s r_s^3
    DF (mass per phase-space volume):
                             f_s = M_s / (r_s^3 V_s^3) = 1 / (G^{3/2} ρ_s^{1/2} r_s^3)

so the dimensionless DF is

    f̂ = G^{3/2} r_s^3 ρ_s^{1/2} f.

Two halos whose density profiles are related by a similarity transform have the same f̂
once their energies are put on a common footing. Mapping halo i onto a reference halo's
units instead of fully dimensionless ones,

    f_{i→ref} = f_i (r_{s,i}/r_{s,ref})^3 (ρ_{s,i}/ρ_{s,ref})^{1/2},

which is the same curve re-dimensionalised in the reference halo's units.

Note on the abscissa: every curve here, stored or returned, uses ε̂ = E/Ψ_i(0). It pins both
endpoints for every halo (0 = escape, 1 = potential centre), so the cliff at ε̂ → 1 aligns
across the sample. Ψ_i(0) is a halo-dependent energy scale set by the potential depth (and
hence by the inner grid radius and total mass) rather than by the structure of the profile,
which is a caveat to state rather than a reason to use a different abscissa.
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
