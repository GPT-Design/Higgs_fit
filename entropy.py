# version 0.4
"""
Entropy helper functions and profile registry for the α‑aware Higgs fit.
"""
from typing import Callable, Dict
import numpy as np

# -----------------------------------------------------------------------------
# Simple scalar helper
# -----------------------------------------------------------------------------

def effective_entropy(S_scalar: float, alpha: float) -> float:
    """Return the constant product α × S₀ (useful for quick checks)."""
    return alpha * S_scalar


# -----------------------------------------------------------------------------
# Energy‑dependent entropy profiles
# -----------------------------------------------------------------------------

def constant_entropy(E: np.ndarray, alpha: float, S0: float, **__) -> np.ndarray:
    """Energy‑independent baseline: S_eff(E) = α · S₀."""
    return alpha * S0 * np.ones_like(E)


def powerlaw_entropy(
    E: np.ndarray,
    alpha: float,
    S0: float,
    m_H: float,
    beta: float = 1.0,
    **__,
) -> np.ndarray:
    """Power‑law scaling: S_eff(E) = α · S₀ · (E / m_H)^β."""
    # Validate inputs to avoid domain errors
    if m_H <= 0:
        raise ValueError(f"m_H must be positive, got {m_H}")
    
    if np.any(E <= 0):
        raise ValueError("All energy values must be positive")
    
    if not np.all(np.isfinite(E)):
        raise ValueError("All energy values must be finite")
    
    return alpha * S0 * (E / m_H) ** beta


def log_running_entropy(
    E: np.ndarray,
    alpha: float,
    S0: float,
    m_H: float,
    c: float = 0.0,
    **__,
) -> np.ndarray:
    """Log‑running form: S_eff(E) = α · S₀ · [1 + c · log(E / m_H)]."""
    # Validate inputs to avoid log domain errors
    if m_H <= 0:
        raise ValueError(f"m_H must be positive, got {m_H}")
    
    if np.any(E <= 0):
        raise ValueError("All energy values must be positive")
    
    if not np.all(np.isfinite(E)):
        raise ValueError("All energy values must be finite")
    
    return alpha * S0 * (1.0 + c * np.log(E / m_H))


# -----------------------------------------------------------------------------
# Registry utilities
# -----------------------------------------------------------------------------

_ENTROPY_REGISTRY: Dict[str, Callable] = {
    "constant": constant_entropy,
    "powerlaw": powerlaw_entropy,
    "log": log_running_entropy,
}


def get_entropy_fn(name: str) -> Callable:
    """Return an entropy profile by *name*."""
    try:
        return _ENTROPY_REGISTRY[name]
    except KeyError as err:
        valid = ", ".join(_ENTROPY_REGISTRY.keys())
        raise ValueError(
            f"Unknown entropy profile '{name}'. Choose from: {valid}"
        ) from err


# -----------------------------------------------------------------------------
# Placeholder for future map‑based entropy fields
# -----------------------------------------------------------------------------

def load_entropy_map(filepath: str) -> dict:  # pragma: no cover
    """Stub for loading external entropy lookup tables (not implemented)."""
    raise NotImplementedError("Entropy map loading not implemented yet.")

# end of entropy.py
