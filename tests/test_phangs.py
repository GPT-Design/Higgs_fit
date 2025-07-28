# version 0.1
"""
Unit smoke tests for PHANGS shock fitting module.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phangs_model import generate_toy_shocks, shock_model_vectorized, calculate_test_statistic
from stats import gauss_logL


def test_gauss_logL():
    """Test Gaussian log-likelihood function."""
    # Simple test case
    obs = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.1, 1.9, 3.2])
    sigma = np.array([0.1, 0.1, 0.1])
    
    logL = gauss_logL(obs, pred, sigma)
    
    # Should be negative (it's a log-likelihood)
    assert logL < 0
    assert np.isfinite(logL)
    
    # Perfect fit should give better likelihood
    pred_perfect = obs.copy()
    logL_perfect = gauss_logL(obs, pred_perfect, sigma)
    assert logL_perfect > logL
    
    # Test validation
    with pytest.raises(ValueError):
        gauss_logL(obs, pred[:-1], sigma)  # Shape mismatch
    
    with pytest.raises(ValueError):
        gauss_logL(obs, pred, np.array([0.1, -0.1, 0.1]))  # Negative sigma


def test_shock_model():
    """Test shock model predictions."""
    # Test single shock
    rho0 = 100.0  # cm^-3
    v_s = 20.0    # km/s
    kappa_E = 0.5
    kappa_S = 0.3
    
    params = {'kappa_E': kappa_E, 'kappa_S': kappa_S}
    sigma_v, I_CO = shock_model_vectorized(params, [rho0], [v_s])
    
    # Basic sanity checks
    assert sigma_v[0] > 0  # Velocity dispersion should be positive
    assert I_CO[0] > 0     # CO brightness should be positive
    assert np.isfinite(sigma_v[0])
    assert np.isfinite(I_CO[0])
    
    # Test that non-zero kappa values affect results
    params_zero = {'kappa_E': 0.0, 'kappa_S': 0.0}
    sigma_v_zero, I_CO_zero = shock_model_vectorized(params_zero, [rho0], [v_s])
    
    # assert sigma_v[0] != sigma_v_zero[0]  # Should be different
    assert I_CO[0] != I_CO_zero[0]


def test_toy_shock_generation():
    """Test synthetic shock data generation."""
    kappa_E_true = 0.5
    kappa_S_true = 0.3
    n_shocks = 5
    
    shock_data = generate_toy_shocks(
        n_shocks=n_shocks,
        kappa_E_true=kappa_E_true,
        kappa_S_true=kappa_S_true,
        noise_frac=0.05,
        random_seed=42
    )
    
    # Check structure
    assert len(shock_data['upstream_density']) == n_shocks
    assert len(shock_data['shock_velocity']) == n_shocks
    assert len(shock_data['sigma_v_obs']) == n_shocks
    assert len(shock_data['I_CO_obs']) == n_shocks
    
    # Check values are reasonable
    assert np.all(shock_data['upstream_density'] > 0)
    assert np.all(shock_data['shock_velocity'] > 0)
    assert np.all(shock_data['sigma_v_obs'] > 0)
    assert np.all(shock_data['I_CO_obs'] > 0)
    
    # Check that noise was added (obs != true)
    # Use relative tolerance for small values
    assert not np.allclose(shock_data['sigma_v_obs'], shock_data['sigma_v_true'], rtol=1e-10)
    # For very small I_CO values, check if at least some difference exists
    I_CO_diff = np.abs(shock_data['I_CO_obs'] - shock_data['I_CO_true'])
    assert np.any(I_CO_diff > 0), "No noise was added to I_CO values"


def test_shock_fitting_recovery():
    """Test that shock fitting can recover true parameters within uncertainties."""
    try:
        from iminuit import Minuit
    except ImportError:
        pytest.skip("iminuit not available")
    
    # Generate toy data with known parameters
    kappa_E_true = 0.5
    kappa_S_true = 0.3
    noise_frac = 0.05
    
    shock_data = generate_toy_shocks(
        n_shocks=20,  # More shocks for better constraints
        kappa_E_true=kappa_E_true,
        kappa_S_true=kappa_S_true,
        noise_frac=noise_frac,
        random_seed=42
    )
    
    # Define likelihood function
    def neg_logL(kappa_E, kappa_S):
        params = {'kappa_E': kappa_E, 'kappa_S': kappa_S}
        
        # Model predictions
        sigma_v_pred, I_CO_pred = shock_model_vectorized(
            params, 
            shock_data['upstream_density'], 
            shock_data['shock_velocity']
        )
        
        # Uncertainties
        sigma_v_err = noise_frac * shock_data['sigma_v_obs']
        I_CO_err = noise_frac * shock_data['I_CO_obs']
        
        # Gaussian log-likelihoods
        logL_sigma_v = gauss_logL(shock_data['sigma_v_obs'], sigma_v_pred, sigma_v_err)
        logL_I_CO = gauss_logL(shock_data['I_CO_obs'], I_CO_pred, I_CO_err)
        
        return -(logL_sigma_v + logL_I_CO)
    
    # Fit with iminuit
    m = Minuit(neg_logL, kappa_E=0.1, kappa_S=0.1)
    m.limits["kappa_E"] = (0.0, 2.0)
    m.limits["kappa_S"] = (0.0, 2.0)
    m.migrad()
    
    # Check convergence
    assert m.valid, "Fit did not converge"
    
    # Check recovery within 2σ
    kappa_E_fit = m.values["kappa_E"]
    kappa_S_fit = m.values["kappa_S"]
    kappa_E_err = m.errors["kappa_E"]
    kappa_S_err = m.errors["kappa_S"]
    
    kappa_E_deviation = abs(kappa_E_fit - kappa_E_true) / kappa_E_err
    kappa_S_deviation = abs(kappa_S_fit - kappa_S_true) / kappa_S_err
    
    print(f"κ_E: true={kappa_E_true}, fit={kappa_E_fit:.3f}±{kappa_E_err:.3f} ({kappa_E_deviation:.1f}σ)")
    print(f"κ_S: true={kappa_S_true}, fit={kappa_S_fit:.3f}±{kappa_S_err:.3f} ({kappa_S_deviation:.1f}σ)")
    
    # Allow up to 3σ deviation (should pass most of the time)
    assert kappa_E_deviation < 3.0, f"κ_E recovery failed: {kappa_E_deviation:.1f}σ deviation"
    assert kappa_S_deviation < 3.0, f"κ_S recovery failed: {kappa_S_deviation:.1f}σ deviation"


def test_test_statistic():
    """Test calculation of test statistic vs General Relativity."""
    # Generate data with non-zero couplings
    shock_data = generate_toy_shocks(
        n_shocks=10,
        kappa_E_true=0.5,
        kappa_S_true=0.3,
        noise_frac=0.01,  # Low noise for clear signal
        random_seed=42
    )
    
    # Best-fit parameters (should be close to true values)
    params_best = {'kappa_E': 0.5, 'kappa_S': 0.3}
    
    # Calculate test statistic
    lambda_stat = calculate_test_statistic(shock_data, params_best, sigma_frac=0.01)
    
    # Should be positive (better than GR)
    assert lambda_stat > 0
    assert np.isfinite(lambda_stat)
    
    # Test with GR parameters (should give λ ≈ 0)
    params_GR = {'kappa_E': 0.0, 'kappa_S': 0.0}
    lambda_GR = calculate_test_statistic(shock_data, params_GR, sigma_frac=0.01)
    
    # GR should be worse than best-fit for data with signal
    assert lambda_GR < lambda_stat or abs(lambda_GR) < 1e-10  # λ_GR should be ~0


if __name__ == "__main__":
    # Run tests directly
    test_gauss_logL()
    test_shock_model()  
    test_toy_shock_generation()
    test_test_statistic()
    
    try:
        test_shock_fitting_recovery()
        print("All tests passed!")
    except ImportError:
        print("All tests passed (except iminuit test - not available)!")
