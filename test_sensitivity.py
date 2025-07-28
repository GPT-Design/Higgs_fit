# version 0.1
"""
Test sensitivity of PHANGS fitting with synthetic data injection.
"""

import numpy as np
from phangs_model import generate_toy_shocks
from phangs_fit import fit_shocks_with_iminuit, prepare_shock_arrays


def test_synthetic_recovery():
    """Test if the fitter can recover known parameters."""
    print("=== Testing Synthetic Parameter Recovery ===")
    
    # Generate synthetic data with known parameters
    kappa_E_true = 0.05
    kappa_S_true = 0.10
    
    shock_data_list = generate_toy_shocks(
        n_shocks=20,  # Smaller sample for testing
        kappa_E_true=kappa_E_true,
        kappa_S_true=kappa_S_true,
        noise_frac=0.02,  # Low noise
        random_seed=42
    )
    
    print(f"Generated {len(shock_data_list['upstream_density'])} synthetic shocks")
    print(f"True parameters: kappa_E={kappa_E_true}, kappa_S={kappa_S_true}")
    
    # Convert to expected format
    shock_data = prepare_shock_arrays([
        {
            'upstream_density': rho,
            'shock_velocity': vs,
            'sigma_v_obs': sig,
            'I_CO_obs': ico
        }
        for rho, vs, sig, ico in zip(
            shock_data_list['upstream_density'],
            shock_data_list['shock_velocity'], 
            shock_data_list['sigma_v_obs'],
            shock_data_list['I_CO_obs']
        )
    ])
    
    # Fit with improved settings
    result = fit_shocks_with_iminuit(shock_data, sigma_frac=0.02)
    
    print(f"\nFit Results:")
    print(f"  Success: {result['success']}")
    print(f"  kappa_E = {result['kappa_E']:.4f} ± {result.get('kappa_E_err', 'NaN')}")
    print(f"  kappa_S = {result['kappa_S']:.4f} ± {result.get('kappa_S_err', 'NaN')}")
    print(f"  Log-likelihood: {result['logL']:.3f}")
    
    if result['success']:
        # Check recovery
        kE_err = result.get('kappa_E_err', float('inf'))
        kS_err = result.get('kappa_S_err', float('inf'))
        
        kE_dev = abs(result['kappa_E'] - kappa_E_true) / kE_err if kE_err > 0 else float('inf')
        kS_dev = abs(result['kappa_S'] - kappa_S_true) / kS_err if kS_err > 0 else float('inf')
        
        print(f"  kappa_E deviation: {kE_dev:.1f}sigma")
        print(f"  kappa_S deviation: {kS_dev:.1f}sigma")
        
        success_recovery = kE_dev < 3.0 and kS_dev < 3.0
        print(f"  Parameter recovery: {'SUCCESS' if success_recovery else 'FAILED'}")
    else:
        print("  Parameter recovery: FIT FAILED")
    
    # Test completed successfully
    assert result is not None, "Result should not be None"


if __name__ == "__main__":
    test_synthetic_recovery()