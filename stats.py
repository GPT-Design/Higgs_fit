# version 0.3
"""
Statistical goodness-of-fit utilities for the Higgs entropy fit.

Includes
--------
* Kolmogorov–Smirnov two-sample test (legacy)
* Pearson χ² test (with tunable ε floor)
* Poisson log-likelihood – the preferred metric for binned counts
"""

from typing import Tuple

import numpy as np
from scipy import stats as _spstats
from scipy.special import gammaln

# -----------------------------------------------------------------------------
# Kolmogorov–Smirnov (legacy 1-D)
# -----------------------------------------------------------------------------

def ks_statistic(observed: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """Return (D, p) from two-sample KS test (SciPy wrapper)."""
    D, p = _spstats.ks_2samp(observed, predicted)
    return float(D), float(p)


# -----------------------------------------------------------------------------
# Pearson χ² (histogram counts or densities)
# -----------------------------------------------------------------------------

def chi2_statistic(
    observed: np.ndarray,
    predicted: np.ndarray,
    ddof: int = 0,
    eps_scale: float = 1e-3,
) -> Tuple[float, float]:
    """Return (χ², p) using a small ε floor to avoid division by zero.

    Args
    ----
    observed / predicted : arrays of same shape (counts or densities)
    ddof                 : number of fitted parameters removed from dof
    eps_scale            : floor as *eps_scale × max(observed, predicted)*
    """
    eps = eps_scale * max(observed.max(), predicted.max())
    chi2 = float(np.sum((observed - predicted) ** 2 / (predicted + eps)))
    dof = int(observed.size - 1 - ddof)
    p_value = float(1.0 - _spstats.chi2.cdf(chi2, dof))
    return chi2, p_value


# -----------------------------------------------------------------------------
# Poisson log-likelihood (recommended)
# -----------------------------------------------------------------------------

def log_poisson_likelihood(
    counts: np.ndarray,
    mu: np.ndarray,
    *,
    eps: float = 1e-9,
) -> float:
    """Return the total log-likelihood: Σ [n log μ – μ – log(n!)].

    Parameters
    ----------
    counts : observed event counts n_i (non-negative integers).
    mu     : expected counts μ_i (positive floats).
    eps    : small floor added to μ to avoid log(0).

    Notes
    -----
    • Uses Stirling's approximation for n ≥ 30:
      log n! ≈ n log n – n + 0.5 log(2πn).
    • For n < 30, uses exact gammaln counts.
    """
    counts = np.asarray(counts, dtype=float)
    mu = np.asarray(mu, dtype=float)
    
    # Validate input arrays
    if counts.shape != mu.shape:
        raise ValueError(f"counts and mu must have same shape: {counts.shape} != {mu.shape}")
    
    if counts.size == 0:
        raise ValueError("Input arrays cannot be empty")
    
    # Validate counts are non-negative
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    
    # Validate mu values are positive
    if np.any(mu <= 0):
        raise ValueError("mu values must be positive")
    
    # Validate all values are finite
    if not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite")
    
    if not np.all(np.isfinite(mu)):
        raise ValueError("mu values must be finite")
    
    mu = mu + eps

    # n log μ – μ
    ll = counts * np.log(mu) - mu

    # –log(n!)
    mask = counts >= 30
    # Stirling approximation for large counts
    ll[mask] -= (
        counts[mask] * np.log(counts[mask])
        - counts[mask]
        + 0.5 * np.log(2 * np.pi * counts[mask])
    )
    # exact for small counts
    ll[~mask] -= gammaln(counts[~mask] + 1)

    return float(ll.sum())

# end of stats.py
