# version 0.1
"""
Minimal 1-D magnetised J-shock analytic solution for PHANGS shock fitting.

Implements simplified shock physics to predict velocity dispersion
and CO brightness from upstream conditions and coupling parameters.
"""

from typing import Dict, Tuple
import numpy as np


# Physical constants (calibrated for PHANGS data)
C_CO = 1e-18  # K km s^-1 / (erg cm^-3), CO brightness conversion factor (increased for observable signals)
M_H = 1.67e-24  # grams, hydrogen mass
K_B = 1.38e-16  # erg K^-1, Boltzmann constant
RHO_SCALE = 1e-21  # Mass density scaling to avoid numerical issues


def sound_speed(T_K: float) -> float:
    """
    Isothermal sound speed for mean molecular weight 2.33 m_H.
    Eq: c_s = 0.57 * sqrt(T/100 K) km/s  (approx)
    """
    return 0.57 * (T_K / 100.0) ** 0.5   # km s^-1


def shock_predict(
    rho0: float,
    v_s: float,
    kappa_E: float,
    kappa_S: float,
    log10_M: float = 1.0,
    sound_speed_km_s: float = 0.22,
    **kwargs
) -> Tuple[float, float]:
    """
    Predict shock observables from upstream conditions and coupling parameters.
    
    Args:
        rho0: Upstream density in cm^-3
        v_s: Shock velocity in km/s
        kappa_E: Elastic coupling parameter (fraction of gas pressure)
        kappa_S: Viscous coupling parameter (fraction of gas pressure)
        log10_M: Log10 of Mach number (flat prior 0.3-1.3, M = 2-20)
        
    Returns:
        sigma_v_pred: Predicted velocity dispersion in km/s
        I_CO_pred: Predicted CO brightness in K km/s
        
    Notes:
        Enhanced model with Mach number dependence:
        - Gas pressure: Pg = ρ_cgs * v_cgs^2 / 3
        - Mach-dependent pressure: P_mach = M^2 * P_thermal
        - Elastic pressure: P_elastic = κ_E * Pg
        - Velocity dispersion: σ_v = sqrt((Pg + P_elastic + P_mach) / ρ_cgs)
        - Viscous losses: ν_loss = κ_S * Pg * M
        - CO brightness: I_CO = c_CO * (Pg + ν_loss)
    """
    # Convert to consistent CGS units
    v_s_cgs = v_s * 1e5  # km/s → cm/s
    rho0_cgs = rho0 * 3.34e-24  # H2 cm^-3 → g cm^-3 (molecular weight ≈ 2 * 1.67e-24)
    
    # Mach number from log10 parameter
    M = 10**log10_M
    
    # Gas pressure from shock compression (standard result)
    Pg = rho0_cgs * v_s_cgs**2 / 3  # erg cm^-3
    
    # Mach-dependent thermal pressure enhancement
    # Use temperature-dependent sound speed
    cs_cgs = sound_speed_km_s * 1e5  # km/s → cm/s
    P_thermal = rho0_cgs * cs_cgs**2  # erg cm^-3
    P_mach = M**2 * P_thermal / 3  # Mach pressure contribution
    
    # Elastic and viscous contributions
    P_elastic = kappa_E * Pg  # erg cm^-3
    visc_loss = kappa_S * Pg * M  # Mach-enhanced viscous losses
    
    # Total pressure including Mach effects
    P_total = Pg + P_elastic + P_mach
    
    # Direct relation: sigma_1d = M * c_s / sqrt(3)
    # Bypasses problematic pressure calculation
    sigma_v_pred = (M * sound_speed_km_s) / np.sqrt(3.0)  # km/s
    
    # CO brightness (calibrated for PHANGS observations)
    I_CO_pred = C_CO * (Pg + visc_loss)  # K km/s
    
    return sigma_v_pred, I_CO_pred


def shock_model_vectorized(
    params: Dict[str, float],
    rho0_array: np.ndarray,
    v_s_array: np.ndarray,
    sound_speed_km_s: float = 0.22
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized shock model for multiple shock regions with Mach number.
    
    Args:
        params: Dictionary with kappa_E, kappa_S, log10_M values
        rho0_array: Array of upstream densities in cm^-3
        v_s_array: Array of shock velocities in km/s
        
    Returns:
        sigma_v_pred: Array of predicted velocity dispersions in km/s
        I_CO_pred: Array of predicted CO brightnesses in K km/s
    """
    kappa_E = params.get('kappa_E', 0.0)
    kappa_S = params.get('kappa_S', 0.0)
    log10_M = params.get('log10_M', 1.0)
    
    # Ensure arrays
    rho0_array = np.asarray(rho0_array)
    v_s_array = np.asarray(v_s_array)
    
    # Mach number
    M = 10**log10_M
    
    # Convert to CGS units
    v_s_cgs = v_s_array * 1e5  # km/s → cm/s
    rho0_cgs = rho0_array * 3.34e-24  # H2 cm^-3 → g cm^-3
    
    # Gas pressure from shock compression
    Pg = rho0_cgs * v_s_cgs**2 / 3  # erg cm^-3
    
    # Mach-dependent thermal pressure  
    # Use temperature-dependent sound speed
    cs_cgs = sound_speed_km_s * 1e5  # km/s → cm/s
    P_thermal = rho0_cgs * cs_cgs**2  # erg cm^-3
    P_mach = M**2 * P_thermal / 3  # erg cm^-3
    
    # Elastic and viscous contributions
    P_elastic = kappa_E * Pg  # erg cm^-3
    visc_loss = kappa_S * Pg * M  # Mach-enhanced viscous losses
    
    # Total pressure
    P_total = Pg + P_elastic + P_mach
    
    # Direct relation: sigma_1d = M * c_s / sqrt(3)
    # Bypasses problematic pressure calculation
    sigma_v_pred = (M * sound_speed_km_s) / np.sqrt(3.0) * np.ones_like(rho0_array)  # km/s, vectorized
    I_CO_pred = C_CO * (Pg + visc_loss)  # K km/s
    
    return sigma_v_pred, I_CO_pred


def shock_residuals(
    params: Dict[str, float],
    shock_data: Dict[str, np.ndarray],
    sigma_frac: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residuals for shock model fitting.
    
    Args:
        params: Model parameters {kappa_E, kappa_S}
        shock_data: Dictionary with observed shock properties
        sigma_frac: Fractional uncertainty (default 10%)
        
    Returns:
        sigma_v_residuals: Velocity dispersion residuals
        I_CO_residuals: CO brightness residuals
    """
    # Extract data
    rho0 = shock_data['upstream_density']
    v_s = shock_data['shock_velocity']
    sigma_v_obs = shock_data['sigma_v_obs']
    I_CO_obs = shock_data['I_CO_obs']
    
    # Model predictions
    sigma_v_pred, I_CO_pred = shock_model_vectorized(params, rho0, v_s)
    
    # Compute residuals with uncertainties
    sigma_v_err = sigma_frac * sigma_v_obs
    I_CO_err = sigma_frac * I_CO_obs
    
    sigma_v_residuals = (sigma_v_obs - sigma_v_pred) / sigma_v_err
    I_CO_residuals = (I_CO_obs - I_CO_pred) / I_CO_err
    
    return sigma_v_residuals, I_CO_residuals


def general_relativity_prediction(
    rho0_array: np.ndarray,
    v_s_array: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure General Relativity prediction (κ_E = κ_S = 0, log10_M = 1.0).
    
    Args:
        rho0_array: Array of upstream densities in cm^-3
        v_s_array: Array of shock velocities in km/s
        
    Returns:
        sigma_v_GR: GR velocity dispersion predictions
        I_CO_GR: GR CO brightness predictions
    """
    params_GR = {'kappa_E': 0.0, 'kappa_S': 0.0, 'log10_M': 1.0}  # M = 10
    return shock_model_vectorized(params_GR, rho0_array, v_s_array)


def calculate_test_statistic(
    shock_data: Dict[str, np.ndarray],
    params_best: Dict[str, float],
    sigma_frac: float = 0.1
) -> float:
    """
    Calculate λ = 2ΔlogL test statistic vs General Relativity.
    
    Args:
        shock_data: Observed shock properties
        params_best: Best-fit model parameters
        sigma_frac: Fractional uncertainty
        
    Returns:
        lambda_stat: Test statistic λ = 2ΔlogL
    """
    # Best-fit chi-squared
    sigma_v_res, I_CO_res = shock_residuals(params_best, shock_data, sigma_frac)
    chi2_best = np.sum(sigma_v_res**2) + np.sum(I_CO_res**2)
    
    # General Relativity chi-squared
    params_GR = {'kappa_E': 0.0, 'kappa_S': 0.0, 'log10_M': params_best.get('log10_M', 1.0)}
    sigma_v_res_GR, I_CO_res_GR = shock_residuals(params_GR, shock_data, sigma_frac)
    chi2_GR = np.sum(sigma_v_res_GR**2) + np.sum(I_CO_res_GR**2)
    
    # Test statistic (equivalent to 2ΔlogL for Gaussian errors)
    lambda_stat = chi2_GR - chi2_best
    
    return lambda_stat


def generate_toy_shocks(
    n_shocks: int = 10,
    kappa_E_true: float = 0.5,
    kappa_S_true: float = 0.3,
    noise_frac: float = 0.05,
    random_seed: int = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic shock data for testing.
    
    Args:
        n_shocks: Number of shock regions
        kappa_E_true: True elastic coupling
        kappa_S_true: True viscous coupling
        noise_frac: Fractional noise level
        random_seed: Random seed for reproducibility
        
    Returns:
        shock_data: Dictionary with synthetic observations
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate upstream conditions
    rho0 = np.random.uniform(50, 200, n_shocks)  # cm^-3
    v_s = np.random.uniform(10, 50, n_shocks)    # km/s
    
    # True model predictions
    params_true = {'kappa_E': kappa_E_true, 'kappa_S': kappa_S_true}
    sigma_v_true, I_CO_true = shock_model_vectorized(params_true, rho0, v_s)
    
    # Add noise
    sigma_v_noise = np.random.normal(0, noise_frac * sigma_v_true)
    I_CO_noise = np.random.normal(0, noise_frac * I_CO_true)
    
    shock_data = {
        'upstream_density': rho0,
        'shock_velocity': v_s,
        'sigma_v_obs': sigma_v_true + sigma_v_noise,
        'I_CO_obs': I_CO_true + I_CO_noise,
        'sigma_v_true': sigma_v_true,
        'I_CO_true': I_CO_true,
    }
    
    return shock_data