# version 0.1
"""
CLI interface for PHANGS shock fitting using iminuit optimization.

Usage: python phangs_fit.py data/NGC4321_cube.fits --mask mask.fits
"""

import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import csv
from datetime import datetime

from phangs_loader import load_phangs_data
from phangs_model import shock_model_vectorized, general_relativity_prediction, calculate_test_statistic, sound_speed
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
    
    def __init__(self, shock_data: Dict[str, np.ndarray], sigma_frac: float = 0.1, sound_speed_km_s: float = 0.22):
        self.shock_data = shock_data
        self.sigma_frac = sigma_frac
        self.sound_speed_km_s = sound_speed_km_s
        
        # Extract data
        self.rho0 = shock_data['upstream_density']
        self.v_s = shock_data['shock_velocity']
        self.sigma_v_obs = shock_data['sigma_v_obs']
        self.I_CO_obs = shock_data['I_CO_obs']
        
        # Uncertainties (smaller, data-driven)
        self.sigma_v_err = np.maximum(0.05 * self.sigma_v_obs, 1.0)  # 5% fractional, 1.0 km/s floor
        self.I_CO_err = np.maximum(0.02 * np.abs(self.I_CO_obs), 0.1)  # 2% fractional, 0.1 K km/s floor
    
    def __call__(self, kappa_E: float, kappa_S: float, log10_M: float) -> float:
        """Return negative log-likelihood for minimization."""
        params = {'kappa_E': kappa_E, 'kappa_S': kappa_S, 'log10_M': log10_M}
        
        # Model predictions
        sigma_v_pred, I_CO_pred = shock_model_vectorized(params, self.rho0, self.v_s, self.sound_speed_km_s)
        
        # Gaussian log-likelihoods
        logL_sigma_v = gauss_logL(self.sigma_v_obs, sigma_v_pred, self.sigma_v_err)
        logL_I_CO = gauss_logL(self.I_CO_obs, I_CO_pred, self.I_CO_err)
        
        # Return negative for minimization
        return -(logL_sigma_v + logL_I_CO)


def fit_shocks_with_iminuit(shock_data: Dict[str, np.ndarray], 
                           sigma_frac: float = 0.1,
                           sound_speed: float = 0.22,
                           fix_kappa: bool = False) -> Dict[str, Any]:
    """Fit shock model using iminuit optimization."""
    try:
        from iminuit import Minuit
    except ImportError:
        raise ImportError("iminuit required for shock fitting. Install with: pip install iminuit")
    
    # Create likelihood function
    likelihood = ShockLikelihood(shock_data, sigma_frac, sound_speed)
    
    # Calculate initial Mach number estimate from median σ/cs
    median_sigma = np.median(shock_data['sigma_v_obs'])
    mach_estimate = median_sigma / sound_speed
    log10_M_init = np.log10(max(1.0, min(100.0, mach_estimate)))  # Clamp to [1, 100]
    
    print(f"  Initial Mach estimate: sigma={median_sigma:.1f}/cs={sound_speed:.2f} = M~{mach_estimate:.1f} (log10_M={log10_M_init:.2f})")
    
    # Set up Minuit with 3 parameters including Mach number  
    # Use exact specifications: limits, init, steps
    if fix_kappa:
        print("  DIAGNOSTIC MODE: Fixing kappa terms to zero for convergence test")
        m = Minuit(likelihood, kappa_E=0.0, kappa_S=0.0, log10_M=log10_M_init)
        m.fixed["kappa_E"] = True
        m.fixed["kappa_S"] = True
    else:
        m = Minuit(likelihood, kappa_E=0.02, kappa_S=0.02, log10_M=log10_M_init)
        m.limits["kappa_E"] = (0.0, 0.5)  # Exact specification
        m.limits["kappa_S"] = (0.0, 0.5)  # Exact specification
        m.errors["kappa_E"] = 0.01  # Exact specification
        m.errors["kappa_S"] = 0.01  # Exact specification
    
    # Mach bounds: M = 1-100 (log10_M: 0-2)
    m.limits["log10_M"] = (0.0, 2.0)  # Exact specification
    m.errors["log10_M"] = 0.05   # Larger step size as specified
    
    # Perform fit
    m.migrad()
    
    # Extract results
    result = {
        'method': 'iminuit',
        'success': m.valid,
        'kappa_E': m.values["kappa_E"],
        'kappa_S': m.values["kappa_S"],  
        'log10_M': m.values["log10_M"],
        'Mach': 10**m.values["log10_M"],  # Also store actual Mach number
        'kappa_E_err': m.errors["kappa_E"] if m.valid else np.nan,
        'kappa_S_err': m.errors["kappa_S"] if m.valid else np.nan,
        'log10_M_err': m.errors["log10_M"] if m.valid else np.nan,
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
        result['log10_M_minos_lower'] = m.merrors["log10_M"].lower
        result['log10_M_minos_upper'] = m.merrors["log10_M"].upper
    except:
        pass
    
    # Calculate test statistic vs GR  
    params_best = {'kappa_E': result['kappa_E'], 'kappa_S': result['kappa_S'], 'log10_M': result['log10_M']}
    lambda_stat = calculate_test_statistic(shock_data, params_best, sigma_frac)
    result['lambda_vs_GR'] = lambda_stat
    
    return result


def save_results_txt(result: Dict[str, Any], output_file: Path, metadata: Dict[str, Any] = None):
    """Save results in human-readable TXT format."""
    from scipy.stats import chi2
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PHANGS Shock Fitting Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Metadata
        if metadata:
            f.write("Data Information:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of shock regions: {metadata.get('n_shocks', 'N/A')}\n")
            f.write(f"Cube file: {metadata.get('cube_file', 'N/A')}\n")
            f.write(f"Mask file: {metadata.get('mask_file', 'N/A')}\n")
            f.write(f"Velocity threshold: {metadata.get('threshold', 'N/A')} km/s\n")
            f.write(f"Fractional uncertainty: {metadata.get('sigma_frac', 'N/A')}\n\n")
        
        # Fit results
        f.write("Fit Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Method: {result.get('method', 'N/A')}\n")
        f.write(f"Convergence: {'Success' if result.get('success', False) else 'Failed'}\n")
        f.write(f"Function evaluations: {result.get('nfev', 'N/A')}\n\n")
        
        if result.get('success', False):
            # Best-fit parameters
            f.write("Best-fit Parameters:\n")
            f.write("-" * 20 + "\n")
            f.write(f"kappa_E = {result['kappa_E']:.6f} ± {result.get('kappa_E_err', np.nan):.6f}\n")
            f.write(f"kappa_S = {result['kappa_S']:.6f} ± {result.get('kappa_S_err', np.nan):.6f}\n")
            f.write(f"log10_M = {result['log10_M']:.4f} ± {result.get('log10_M_err', np.nan):.4f}\n")
            f.write(f"Mach number = {result.get('Mach', 10**result['log10_M']):.2f}\n\n")
            
            # MINOS errors if available
            if 'kappa_E_minos_lower' in result:
                f.write("MINOS Asymmetric Errors:\n")
                f.write("-" * 25 + "\n")
                f.write(f"kappa_E: {result['kappa_E']:.6f} "
                        f"+{result['kappa_E_minos_upper']:.6f} "
                        f"{result['kappa_E_minos_lower']:.6f}\n")
                f.write(f"kappa_S: {result['kappa_S']:.6f} "
                        f"+{result['kappa_S_minos_upper']:.6f} "
                        f"{result['kappa_S_minos_lower']:.6f}\n")
                f.write(f"log10_M: {result['log10_M']:.4f} "
                        f"+{result['log10_M_minos_upper']:.4f} "
                        f"{result['log10_M_minos_lower']:.4f}\n\n")
            
            # Likelihood and statistics
            f.write("Likelihood and Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Log-likelihood: {result.get('logL', np.nan):.3f}\n")
            f.write(f"Negative log-likelihood: {result.get('neg_logL', np.nan):.3f}\n")
            
            # Test statistic vs GR
            lambda_val = result.get('lambda_vs_GR', 0)
            f.write(f"Test statistic (2*DeltaLogL vs GR): {lambda_val:.6f}\n")
            
            # Significance assessment
            if lambda_val > 0:
                p_value = 1 - chi2.cdf(lambda_val, df=3)
                sigma_equivalent = np.sqrt(chi2.ppf(1 - p_value/2, df=1)) if p_value > 0 else np.inf
                f.write(f"p-value vs GR: {p_value:.6f}\n")
                f.write(f"Significance: {sigma_equivalent:.2f} sigma\n")
            
        else:
            f.write("Fit did not converge successfully.\n")
            f.write(f"Final parameters: kappa_E={result.get('kappa_E', np.nan):.6f}, "
                    f"kappa_S={result.get('kappa_S', np.nan):.6f}, "
                    f"log10_M={result.get('log10_M', np.nan):.4f}\n")


def save_results_csv(result: Dict[str, Any], output_file: Path, metadata: Dict[str, Any] = None):
    """Save results in CSV format for spreadsheet analysis."""
    from scipy.stats import chi2
    
    # Prepare data for CSV
    csv_data = []
    
    # Header row with all possible columns
    headers = [
        'timestamp', 'cube_file', 'mask_file', 'n_shocks', 'threshold', 'sigma_frac',
        'method', 'success', 'nfev', 
        'kappa_E', 'kappa_E_err', 'kappa_E_minos_lower', 'kappa_E_minos_upper',
        'kappa_S', 'kappa_S_err', 'kappa_S_minos_lower', 'kappa_S_minos_upper',
        'log10_M', 'log10_M_err', 'log10_M_minos_lower', 'log10_M_minos_upper',
        'Mach', 'logL', 'neg_logL', 'lambda_vs_GR', 'p_value_vs_GR', 'sigma_equivalent'
    ]
    
    # Calculate additional statistics
    lambda_val = result.get('lambda_vs_GR', 0)
    p_value = 1 - chi2.cdf(lambda_val, df=3) if lambda_val > 0 else np.nan
    sigma_equiv = np.sqrt(chi2.ppf(1 - p_value/2, df=1)) if p_value > 0 and not np.isnan(p_value) else np.nan
    
    # Data row
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metadata.get('cube_file', '') if metadata else '',
        metadata.get('mask_file', '') if metadata else '',
        metadata.get('n_shocks', '') if metadata else '',
        metadata.get('threshold', '') if metadata else '',
        metadata.get('sigma_frac', '') if metadata else '',
        result.get('method', ''),
        result.get('success', False),
        result.get('nfev', ''),
        result.get('kappa_E', np.nan),
        result.get('kappa_E_err', np.nan),
        result.get('kappa_E_minos_lower', ''),
        result.get('kappa_E_minos_upper', ''),
        result.get('kappa_S', np.nan),
        result.get('kappa_S_err', np.nan),
        result.get('kappa_S_minos_lower', ''),
        result.get('kappa_S_minos_upper', ''),
        result.get('log10_M', np.nan),
        result.get('log10_M_err', np.nan),
        result.get('log10_M_minos_lower', ''),
        result.get('log10_M_minos_upper', ''),
        result.get('Mach', np.nan),
        result.get('logL', np.nan),
        result.get('neg_logL', np.nan),
        lambda_val,
        p_value,
        sigma_equiv
    ]
    
    # Write CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(row)


def print_fit_results(result: Dict[str, Any], n_shocks: int):
    """Print formatted fit results."""
    print(f"\nShock Fitting Results ({n_shocks} shock regions)")
    print("=" * 50)
    
    if result['success']:
        print(f"kappa_E = {result['kappa_E']:.4f} ± {result['kappa_E_err']:.4f}")
        print(f"kappa_S = {result['kappa_S']:.4f} ± {result['kappa_S_err']:.4f}")
        print(f"log10_M = {result['log10_M']:.3f} ± {result['log10_M_err']:.3f}")
        print(f"Mach = {result['Mach']:.1f}")
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
            print(f"log10_M MINOS: {result['log10_M']:.3f} "
                  f"+{result['log10_M_minos_upper']:.3f} "
                  f"{result['log10_M_minos_lower']:.3f}")
        
        # Test statistic
        lambda_val = result.get('lambda_vs_GR', 0)
        print(f"lambda = 2*DeltalogL vs GR: {lambda_val:.3f}")
        
        # Significance assessment
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lambda_val, df=3)  # 3 parameters now
        sigma_equivalent = np.sqrt(chi2.ppf(1 - p_value/2, df=1))
        print(f"p-value vs GR: {p_value:.4f} ({sigma_equivalent:.1f}sigma)")
        
    else:
        print("Fit failed to converge!")
        print(f"Final parameters: kappa_E={result['kappa_E']:.4f}, kappa_S={result['kappa_S']:.4f}, log10_M={result['log10_M']:.3f}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Fit PHANGS shock data with magnetised J-shock model"
    )
    
    parser.add_argument("--cube", type=Path, required=True, 
                       help="Path to 3D masked CO cube FITS file (*_strictmask.fits)")
    parser.add_argument("--mask", type=str, default="strict",
                       help="Mask type: 'strict' for automatic detection or path to mask file")
    parser.add_argument("--noise", type=Path, required=True,
                       help="Path to 3D RMS noise cube (*_noise.pbcor.fits)")
    parser.add_argument("--noise2d", action="store_true",
                       help="Use 2D error/weight map instead of 3D noise cube")
    parser.add_argument("--fit-mach", action="store_true", default=True,
                       help="Fit Mach number (log10_M) as third parameter")
    parser.add_argument("--temperature", type=float, default=15.0,
                       help="Gas temperature [K] used for sound speed")
    parser.add_argument("--threshold", type=float, default=0.0,
                       help="Velocity dispersion threshold (km/s) if no mask (default: 0.0, no threshold)")
    parser.add_argument("--min-pixels", type=int, default=5,
                       help="Minimum pixels per shock region")
    parser.add_argument("--sigma-frac", type=float, default=0.1,
                       help="Fractional uncertainty (default 10%%)")
    parser.add_argument("--output", type=Path, default=Path("results"),
                       help="Output directory (default: results)")
    parser.add_argument("--prefix", type=str, default="shock_fit_results",
                       help="Output file prefix (default: shock_fit_results)")
    parser.add_argument("--formats", nargs='+', choices=['json', 'txt', 'csv'], 
                       default=['json', 'txt', 'csv'],
                       help="Output formats (default: all formats)")
    parser.add_argument("--dump-sigma-hist", action="store_true",
                       help="Save velocity dispersion histogram for literature comparison")
    parser.add_argument("--max-regions", type=int, default=None,
                       help="Limit number of shock regions for testing (default: use all)")
    parser.add_argument("--fix-kappa", action="store_true",
                       help="Freeze kappa terms to zero for first pass convergence test")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cube.exists():
        raise FileNotFoundError(f"Cube file not found: {args.cube}")
    
    if not args.noise.exists():
        raise FileNotFoundError(f"Noise file not found: {args.noise}")
    
    # Handle mask detection
    mask_file = None
    if args.mask == "strict":
        # Automatic detection: cube should be *_strictmask.fits, look for corresponding mask
        cube_name = args.cube.name
        if "_strictmask.fits" in cube_name:
            # The cube IS already the masked cube - no separate mask needed
            print("Using masked cube directly (no separate mask file needed)")
            mask_file = None
        else:
            raise ValueError(f"Expected cube name to contain '_strictmask.fits', got: {cube_name}")
    else:
        # Explicit mask file path
        mask_file = Path(args.mask)
        if not mask_file.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_file}")
    
    # Load data
    print("Loading PHANGS data...")
    # Use the threshold as specified by user (default 0.0 = no threshold)
    shocks, metadata = load_phangs_data(
        args.cube, 
        mask_file, 
        args.threshold, 
        args.min_pixels,
        noise_file=args.noise
    )
    
    if len(shocks) == 0:
        print("No shock regions found!")
        return
    
    # Prepare data for fitting
    shock_data = prepare_shock_arrays(shocks)
    
    # Limit regions for testing if requested
    if args.max_regions is not None and len(shocks) > args.max_regions:
        print(f"Limiting to first {args.max_regions} shock regions for testing")
        # Take first N regions
        for key in shock_data:
            shock_data[key] = shock_data[key][:args.max_regions]
    
    # Calculate sound speed for temperature-aware analysis
    cs = sound_speed(args.temperature)
    print(f"Using gas temperature: {args.temperature} K (sound speed: {cs:.2f} km/s)")
    
    print(f"Shock statistics:")
    print(f"  Velocity dispersion: {np.mean(shock_data['sigma_v_obs']):.1f} ± "
          f"{np.std(shock_data['sigma_v_obs']):.1f} km/s")
    print(f"  Median sigma_v: {np.median(shock_data['sigma_v_obs']):.1f} km/s")
    print(f"  Range: {np.min(shock_data['sigma_v_obs']):.1f} - {np.max(shock_data['sigma_v_obs']):.1f} km/s")
    print(f"  CO brightness: {np.mean(shock_data['I_CO_obs']):.3f} ± "
          f"{np.std(shock_data['I_CO_obs']):.3f} K km/s")
    
    # Literature comparison for CO velocity dispersion
    if np.median(shock_data['sigma_v_obs']) > 0:
        literature_range = (6.0, 15.0)  # km/s from literature
        median_sigma = np.median(shock_data['sigma_v_obs'])
        if literature_range[0] <= median_sigma <= literature_range[1]:
            print(f"  OK: Median sigma_v within literature range {literature_range[0]}-{literature_range[1]} km/s")
        else:
            print(f"  WARNING: Median sigma_v outside literature range {literature_range[0]}-{literature_range[1]} km/s")
    
    # Perform fitting
    n_regions = len(shock_data['sigma_v_obs'])
    print(f"\nFitting {n_regions} shock regions...")
    result = fit_shocks_with_iminuit(shock_data, args.sigma_frac, sound_speed=cs, fix_kappa=args.fix_kappa)
    
    # Print results
    print_fit_results(result, len(shocks))
    
    # Save results in multiple formats
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'n_shocks': len(shocks),
        'cube_file': str(args.cube),
        'mask_file': str(mask_file) if mask_file else "strict (embedded in cube)",
        'noise_file': str(args.noise),
        'noise2d': args.noise2d,
        'fit_mach': args.fit_mach,
        'temperature': args.temperature,
        'sound_speed': cs,
        'threshold': args.threshold,
        'sigma_frac': args.sigma_frac,
    }
    
    saved_files = []
    
    # Save velocity dispersion histogram if requested
    if args.dump_sigma_hist:
        sigma_hist_file = args.output / f"{args.prefix}_sigma_histogram.txt"
        sigma_values = shock_data['sigma_v_obs']
        
        with open(sigma_hist_file, 'w') as f:
            f.write("# Velocity Dispersion Histogram Analysis\n")
            f.write(f"# Galaxy: {args.prefix}\n")
            f.write(f"# Temperature: {args.temperature} K (cs = {cs:.2f} km/s)\n")
            f.write(f"# Number of shock regions: {len(sigma_values)}\n")
            f.write("#\n")
            f.write("# Literature comparison: CO velocity dispersion typically 6-15 km/s\n")
            f.write(f"# Mean: {np.mean(sigma_values):.2f} km/s\n")
            f.write(f"# Median: {np.median(sigma_values):.2f} km/s\n")
            f.write(f"# Std: {np.std(sigma_values):.2f} km/s\n")
            f.write(f"# Min: {np.min(sigma_values):.2f} km/s\n")
            f.write(f"# Max: {np.max(sigma_values):.2f} km/s\n")
            f.write("#\n")
            f.write("# Histogram bins (center, count, fraction)\n")
            f.write("# sigma_v_center[km/s]  count  fraction\n")
            
            # Create histogram
            bins = np.linspace(0, max(30, np.max(sigma_values) + 2), 31)  # 1 km/s bins up to 30 or max+2
            hist, bin_edges = np.histogram(sigma_values, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            for center, count in zip(bin_centers, hist):
                fraction = count / len(sigma_values)
                f.write(f"{center:8.1f}        {count:5d}  {fraction:8.4f}\n")
            
            # Add individual values
            f.write("#\n# Individual shock region sigma_v values:\n")
            for i, sigma in enumerate(sigma_values):
                f.write(f"# Region {i+1:3d}: {sigma:.2f} km/s\n")
        
        saved_files.append(str(sigma_hist_file))
        print(f"Velocity dispersion histogram saved to: {sigma_hist_file}")
    
    # Save JSON format (original)
    if 'json' in args.formats:
        import json
        result_file = args.output / f"{args.prefix}.json"
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
            
            json_result['metadata'] = metadata
            json.dump(json_result, f, indent=2)
        saved_files.append(str(result_file))
    
    # Save TXT format (human-readable)
    if 'txt' in args.formats:
        txt_file = args.output / f"{args.prefix}.txt"
        save_results_txt(result, txt_file, metadata)
        saved_files.append(str(txt_file))
    
    # Save CSV format (spreadsheet-friendly)
    if 'csv' in args.formats:
        csv_file = args.output / f"{args.prefix}.csv"
        save_results_csv(result, csv_file, metadata)
        saved_files.append(str(csv_file))
    
    print(f"\nResults saved to:")
    for file_path in saved_files:
        print(f"  - {file_path}")


if __name__ == "__main__":
    main()