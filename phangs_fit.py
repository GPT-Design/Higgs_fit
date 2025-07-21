# version 0.1
"""
CLI interface for PHANGS shock fitting using iminuit optimization.

Usage: python phangs_fit.py data/NGC4321_cube.fits --mask mask.fits
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np

from phangs_loader import load_phangs_data
from phangs_model import shock_model_vectorized, general_relativity_prediction, calculate_test_statistic
from stats import gauss_logL


def prepare_shock_arrays(shocks: list) -> Dict[str, np.ndarray]:
    """Convert list of shock dictionaries to arrays for fitting."""
    shock_data = {
        'upstream_density': np.array([s['upstream_density'] for s in shocks]),
        'shock_velocity': np.array([s['shock_velocity'] for s in shocks]),
        'sigma_v_obs': np.array([s['sigma_v_obs'] for s in shocks]),
        'I_CO_obs': np.array([s['I_CO_obs'] for s in shocks]),
    }
    return shock_data


class ShockLikelihood:
    """Likelihood function for shock fitting."""
    
    def __init__(self, shock_data: Dict[str, np.ndarray], sigma_frac: float = 0.1):
        self.shock_data = shock_data
        self.sigma_frac = sigma_frac
        
        # Extract data
        self.rho0 = shock_data['upstream_density']
        self.v_s = shock_data['shock_velocity']
        self.sigma_v_obs = shock_data['sigma_v_obs']
        self.I_CO_obs = shock_data['I_CO_obs']
        
        # Uncertainties (smaller, data-driven)
        self.sigma_v_err = 0.05 * self.sigma_v_obs  # 5% fractional for velocity dispersion
        self.I_CO_err = np.maximum(0.02 * np.abs(self.I_CO_obs), 0.1)  # 2% fractional, 0.1 K km/s floor
    
    def __call__(self, kappa_E: float, kappa_S: float) -> float:
        """Return negative log-likelihood for minimization."""
        params = {'kappa_E': kappa_E, 'kappa_S': kappa_S}
        
        # Model predictions
        sigma_v_pred, I_CO_pred = shock_model_vectorized(params, self.rho0, self.v_s)
        
        # Gaussian log-likelihoods
        logL_sigma_v = gauss_logL(self.sigma_v_obs, sigma_v_pred, self.sigma_v_err)
        logL_I_CO = gauss_logL(self.I_CO_obs, I_CO_pred, self.I_CO_err)
        
        # Return negative for minimization
        return -(logL_sigma_v + logL_I_CO)


def fit_shocks_with_iminuit(shock_data: Dict[str, np.ndarray], 
                           sigma_frac: float = 0.1) -> Dict[str, Any]:
    """Fit shock model using iminuit optimization."""
    try:
        from iminuit import Minuit
    except ImportError:
        raise ImportError("iminuit required for shock fitting. Install with: pip install iminuit")
    
    # Create likelihood function
    likelihood = ShockLikelihood(shock_data, sigma_frac)
    
    # Set up Minuit with tighter parameter space & better seeds
    m = Minuit(likelihood, kappa_E=0.01, kappa_S=0.05)
    m.limits["kappa_E"] = (0.0, 0.2)  # Tighter limits
    m.limits["kappa_S"] = (0.0, 0.5)  # Tighter limits
    m.errors["kappa_E"] = 0.001  # Smaller step sizes
    m.errors["kappa_S"] = 0.005
    
    # Perform fit
    m.migrad()
    
    # Extract results
    result = {
        'method': 'iminuit',
        'success': m.valid,
        'kappa_E': m.values["kappa_E"],
        'kappa_S': m.values["kappa_S"],
        'kappa_E_err': m.errors["kappa_E"] if m.valid else np.nan,
        'kappa_S_err': m.errors["kappa_S"] if m.valid else np.nan,
        'neg_logL': m.fval,
        'logL': -m.fval,
        'nfev': m.nfcn,
    }
    
    # Add MINOS errors if available
    try:
        m.minos()
        result['kappa_E_minos_lower'] = m.merrors["kappa_E"].lower
        result['kappa_E_minos_upper'] = m.merrors["kappa_E"].upper
        result['kappa_S_minos_lower'] = m.merrors["kappa_S"].lower
        result['kappa_S_minos_upper'] = m.merrors["kappa_S"].upper
    except:
        pass
    
    # Calculate test statistic vs GR
    params_best = {'kappa_E': result['kappa_E'], 'kappa_S': result['kappa_S']}
    lambda_stat = calculate_test_statistic(shock_data, params_best, sigma_frac)
    result['lambda_vs_GR'] = lambda_stat
    
    return result


def print_fit_results(result: Dict[str, Any], n_shocks: int):
    """Print formatted fit results."""
    print(f"\nShock Fitting Results ({n_shocks} shock regions)")
    print("=" * 50)
    
    if result['success']:
        print(f"kappa_E = {result['kappa_E']:.4f} ± {result['kappa_E_err']:.4f}")
        print(f"kappa_S = {result['kappa_S']:.4f} ± {result['kappa_S_err']:.4f}")
        print(f"Log-likelihood: {result['logL']:.3f}")
        print(f"Function evaluations: {result['nfev']}")
        
        # MINOS errors if available
        if 'kappa_E_minos_lower' in result:
            print(f"kappa_E MINOS: {result['kappa_E']:.4f} "
                  f"+{result['kappa_E_minos_upper']:.4f} "
                  f"{result['kappa_E_minos_lower']:.4f}")
            print(f"kappa_S MINOS: {result['kappa_S']:.4f} "
                  f"+{result['kappa_S_minos_upper']:.4f} "
                  f"{result['kappa_S_minos_lower']:.4f}")
        
        # Test statistic
        lambda_val = result.get('lambda_vs_GR', 0)
        print(f"lambda = 2*DeltalogL vs GR: {lambda_val:.3f}")
        
        # Significance assessment
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lambda_val, df=2)  # 2 parameters
        sigma_equivalent = np.sqrt(chi2.ppf(1 - p_value/2, df=1))
        print(f"p-value vs GR: {p_value:.4f} ({sigma_equivalent:.1f}sigma)")
        
    else:
        print("Fit failed to converge!")
        print(f"Final parameters: kappa_E={result['kappa_E']:.4f}, kappa_S={result['kappa_S']:.4f}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Fit PHANGS shock data with magnetised J-shock model"
    )
    
    parser.add_argument("cube", type=Path, help="Path to CO cube FITS file")
    parser.add_argument("--mask", type=Path, help="Path to shock mask FITS file")
    parser.add_argument("--threshold", type=float, default=15.0,
                       help="Velocity dispersion threshold (km/s) if no mask")
    parser.add_argument("--min-pixels", type=int, default=5,
                       help="Minimum pixels per shock region")
    parser.add_argument("--sigma-frac", type=float, default=0.1,
                       help="Fractional uncertainty (default 10%%)")
    parser.add_argument("--output", type=Path, default=Path("phangs_results"),
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cube.exists():
        raise FileNotFoundError(f"Cube file not found: {args.cube}")
    
    if args.mask and not args.mask.exists():
        raise FileNotFoundError(f"Mask file not found: {args.mask}")
    
    # Load data
    print("Loading PHANGS data...")
    shocks, metadata = load_phangs_data(
        args.cube, 
        args.mask, 
        args.threshold, 
        args.min_pixels
    )
    
    if len(shocks) == 0:
        print("No shock regions found!")
        return
    
    # Prepare data for fitting
    shock_data = prepare_shock_arrays(shocks)
    
    print(f"Shock statistics:")
    print(f"  Velocity dispersion: {np.mean(shock_data['sigma_v_obs']):.1f} ± "
          f"{np.std(shock_data['sigma_v_obs']):.1f} km/s")
    print(f"  CO brightness: {np.mean(shock_data['I_CO_obs']):.3f} ± "
          f"{np.std(shock_data['I_CO_obs']):.3f} K km/s")
    
    # Perform fitting
    print(f"\nFitting {len(shocks)} shock regions...")
    result = fit_shocks_with_iminuit(shock_data, args.sigma_frac)
    
    # Print results
    print_fit_results(result, len(shocks))
    
    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    
    import json
    result_file = args.output / "shock_fit_results.json"
    with open(result_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value)
            else:
                json_result[key] = value
        
        json_result['metadata'] = {
            'n_shocks': len(shocks),
            'cube_file': str(args.cube),
            'mask_file': str(args.mask) if args.mask else None,
            'threshold': args.threshold,
            'sigma_frac': args.sigma_frac,
        }
        
        json.dump(json_result, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")


if __name__ == "__main__":
    main()