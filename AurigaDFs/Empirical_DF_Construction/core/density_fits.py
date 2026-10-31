from __future__ import annotations

import numpy as np


def nfw_shape(r: np.ndarray, rs: float) -> np.ndarray:
    x = np.asarray(r, float) / float(rs)
    return 1.0 / (x * (1.0 + x) ** 2)


def einasto_shape(r: np.ndarray, r2: float, alpha: float) -> np.ndarray:
    r = np.asarray(r, float)
    r2 = float(r2)
    alpha = float(alpha)
    return np.exp(-(2.0 / alpha) * ((r / r2) ** alpha - 1.0))


def fit_nfw_einasto_gridsearch(
    r_mid: np.ndarray,
    rho: np.ndarray,
    counts: np.ndarray,
    *,
    r_fit_min: float = 2.0,
    r_fit_max: float = 300.0,
    min_counts: int = 200,
) -> dict[str, dict[str, float]]:
    r_mid = np.asarray(r_mid, float)
    rho = np.asarray(rho, float)
    counts = np.asarray(counts, float)

    m = np.isfinite(r_mid) & np.isfinite(rho) & (rho > 0) & (counts >= float(min_counts))
    m = m & (r_mid >= float(r_fit_min)) & (r_mid <= float(r_fit_max))
    r_fit = r_mid[m]
    rho_fit = rho[m]
    if r_fit.size < 8:
        return {}

    ln_rho = np.log(rho_fit)
    best: dict[str, dict[str, float]] = {}

    # NFW: grid rs
    rs_grid = np.logspace(np.log10(r_fit.min()), np.log10(r_fit.max()), 80)
    best_sse = np.inf
    best_rs = np.nan
    best_rhos = np.nan
    for rs in rs_grid:
        sh = nfw_shape(r_fit, rs)
        if not np.all(np.isfinite(sh)) or np.any(sh <= 0):
            continue
        ln_rhos = float(np.mean(ln_rho - np.log(sh)))
        pred = np.exp(ln_rhos) * sh
        sse = float(np.mean((ln_rho - np.log(pred)) ** 2))
        if sse < best_sse:
            best_sse, best_rs, best_rhos = sse, float(rs), float(np.exp(ln_rhos))
    best["nfw"] = dict(rs=float(best_rs), rhos=float(best_rhos), sse=float(best_sse))

    # Einasto: grid alpha + r2
    alpha_grid = np.linspace(0.10, 0.35, 26)
    r2_grid = np.logspace(np.log10(r_fit.min()), np.log10(r_fit.max()), 70)
    best_sse = np.inf
    best_alpha = np.nan
    best_r2 = np.nan
    best_rho2 = np.nan
    for alpha in alpha_grid:
        for r2 in r2_grid:
            sh = einasto_shape(r_fit, r2, float(alpha))
            if not np.all(np.isfinite(sh)) or np.any(sh <= 0):
                continue
            ln_rho2 = float(np.mean(ln_rho - np.log(sh)))
            pred = np.exp(ln_rho2) * sh
            sse = float(np.mean((ln_rho - np.log(pred)) ** 2))
            if sse < best_sse:
                best_sse, best_alpha, best_r2, best_rho2 = sse, float(alpha), float(r2), float(np.exp(ln_rho2))
    best["einasto"] = dict(alpha=float(best_alpha), r2=float(best_r2), rho2=float(best_rho2), sse=float(best_sse))

    best["fit_meta"] = dict(
        r_fit_min=float(r_fit_min),
        r_fit_max=float(r_fit_max),
        min_counts=float(min_counts),
        n_fit=int(r_fit.size),
    )
    return best

