"""
Changes of variable applied before symbolic regression.

Each maps eps_hat in (0, 1) onto an unbounded axis, so that both asymptotes -- the inner
behaviour as eps_hat -> 0 and the cliff at eps_hat -> 1 -- become expressible by a
low-complexity analytic form. Without one, a fit has to spend complexity reproducing a
divergence at a fixed abscissa, which symbolic regression does badly.

The ordinate is never transformed: every curve here stays log10 f.

`identity` is included as the control -- fitting eps_hat directly -- so a run can show
what the change of variable actually buys.

The parameterised family ln(eps_hat^gamma / (1 - eps_hat^gamma)) is deliberately absent.
It is `logit` at gamma = 1, and choosing gamma by linearity over the whole sample would
leak held-out halos into a cross-validation fold.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

# The abscissa, as a symbol. Positive and < 1 on the fitted grid, which lets sympy
# simplify logs and roots of eps_hat and 1 - eps_hat without a sign assumption.
EPS = sp.Symbol("eps_hat", positive=True)


def identity(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, float)


def logit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.log(x / (1.0 - x))


def sqrt_ratio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.sqrt(x / (1.0 - x))


def tan_pi(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return np.tan(np.pi * (x - 0.5))


def centered_ratio(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - 0.5) / (x * (1.0 - x))


def log_log_ratio(x: np.ndarray) -> np.ndarray:
    """ln(ln x / ln(1-x)). Both logs are negative on (0,1), so the ratio is positive."""
    x = np.asarray(x, float)
    return np.log(np.log(x) / np.log(1.0 - x))


def _sym_identity() -> sp.Expr:
    return EPS


def _sym_logit() -> sp.Expr:
    return sp.log(EPS / (1 - EPS))


def _sym_sqrt_ratio() -> sp.Expr:
    return sp.sqrt(EPS / (1 - EPS))


def _sym_tan_pi() -> sp.Expr:
    return sp.tan(sp.pi * (EPS - sp.Rational(1, 2)))


def _sym_centered_ratio() -> sp.Expr:
    return (EPS - sp.Rational(1, 2)) / (EPS * (1 - EPS))


def _sym_log_log_ratio() -> sp.Expr:
    return sp.log(sp.log(EPS) / sp.log(1 - EPS))


# name -> (callable, LaTeX for the abscissa)
TRANSFORMS: dict[str, tuple] = {
    "identity":       (identity,       r"$\hat{\varepsilon}$"),
    "logit":          (logit,          r"$\ln\!\left(\frac{\hat{\varepsilon}}{1-\hat{\varepsilon}}\right)$"),
    "sqrt_ratio":     (sqrt_ratio,     r"$\sqrt{\frac{\hat{\varepsilon}}{1-\hat{\varepsilon}}}$"),
    "tan_pi":         (tan_pi,         r"$\tan\!\left[\pi\left(\hat{\varepsilon}-\frac{1}{2}\right)\right]$"),
    "centered_ratio": (centered_ratio, r"$\frac{\hat{\varepsilon}-\frac{1}{2}}{\hat{\varepsilon}(1-\hat{\varepsilon})}$"),
    "log_log_ratio":  (log_log_ratio,  r"$\ln\!\left(\frac{\ln\hat{\varepsilon}}{\ln(1-\hat{\varepsilon})}\right)$"),
}

TRANSFORM_NAMES: list[str] = list(TRANSFORMS)

# name -> the same change of variable as a sympy expression in EPS.
#
# These exist so a fitted expression g(u) can be composed back onto eps_hat: the change
# of variable is part of the model, not a preprocessing step, so the form that has to be
# reported -- and whose complexity has to be counted -- is g(T(eps_hat)), not g(u).
# Kept beside the numpy versions deliberately: if one is edited and the other is not, the
# composed expressions silently stop describing the curves that were actually fitted, so
# test_sympy_matches_numpy guards the pair.
SYMPY_TRANSFORMS: dict[str, sp.Expr] = {
    "identity":       _sym_identity(),
    "logit":          _sym_logit(),
    "sqrt_ratio":     _sym_sqrt_ratio(),
    "tan_pi":         _sym_tan_pi(),
    "centered_ratio": _sym_centered_ratio(),
    "log_log_ratio":  _sym_log_log_ratio(),
}


def apply(name: str, x: np.ndarray) -> np.ndarray:
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; have {TRANSFORM_NAMES}")
    return TRANSFORMS[name][0](x)


def sympy_form(name: str) -> sp.Expr:
    """The change of variable as a sympy expression in EPS."""
    if name not in SYMPY_TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; have {TRANSFORM_NAMES}")
    return SYMPY_TRANSFORMS[name]
