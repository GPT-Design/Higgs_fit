# version 0.1
"""
Physics model registry and implementations for Higgs entropy fitting.
"""
import numpy as np
from typing import Callable, Dict
from entropy import get_entropy_fn

# Registry for available model kernels
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a new physics model under a given name.

    Usage:
        @register_model("my_model")
        def my_model(energy, params):
            ...
    """
    def decorator(fn: Callable) -> Callable:
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model(name: str) -> Callable:
    """
    Retrieve a registered model by name.

    Raises:
        ValueError: If the model name is not found in the registry.
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        valid = ", ".join(MODEL_REGISTRY.keys()) or "<none>"
        raise ValueError(f"Model '{name}' not found. Available: {valid}")


@register_model("breit_wigner_entropy")
def breit_wigner_entropy(energy: np.ndarray, params: dict) -> np.ndarray:
    """
    Relativistic Breit–Wigner line shape with entropy coupling term.

    Args:
        energy (np.ndarray): Array of energy values (GeV).
        params (dict): Dictionary with keys:
            - m_H (float): Higgs mass (GeV).
            - Gamma (float): Higgs width (GeV).
            - N (float): Normalization constant.
            - kappa (float): Entropy coupling coefficient.
            - alpha (float): Entropy scaling factor.
            - S_scalar (float): Base entropy scalar.
            - entropy_shape (str): Entropy profile type ("constant", "powerlaw", "log").
            - c (float): Log-running coefficient (for "log" entropy).
            - beta (float): Power-law exponent (for "powerlaw" entropy).

    Returns:
        np.ndarray: Predicted observable at each energy.
    """
    m_H = params["m_H"]
    Gamma = params["Gamma"]
    N = params.get("N", 1.0)
    kappa = params["kappa"]
    alpha = params["alpha"]
    S_scalar = params["S_scalar"]
    entropy_shape = params.get("entropy_shape", "log")
    c = params.get("c", 0.0)
    beta = params.get("beta", 1.0)

    # Validate parameters
    if Gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {Gamma}")
    if m_H <= 0:
        raise ValueError(f"m_H must be positive, got {m_H}")

    # Relativistic Breit–Wigner core
    bw = N * (Gamma ** 2) / ((energy - m_H) ** 2 + (Gamma ** 2) / 4)

    # Calculate effective entropy using selected profile
    entropy_fn = get_entropy_fn(entropy_shape)
    S_eff = entropy_fn(energy, alpha, S_scalar, m_H=m_H, c=c, beta=beta)

    return bw + kappa * S_eff

# End of models.py
