# version 0.2
"""
Continuous maximization of the Poisson log-likelihood for one data set using
iminuit optimization with fallback coarse grid and optional emcee sampling.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from config import Config
from models import get_model
from entropy import effective_entropy
from stats import log_poisson_likelihood
from file_parser import auto_detect_and_parse
from sweeps import make_counts
from constants import DEFAULT_BINS, DEFAULT_SCALE, DEFAULT_HIGGS_MASS, DEFAULT_HIGGS_WIDTH, DEFAULT_NORMALIZATION


# -----------------------------------------------------------------------------
# Likelihood function for optimization
# -----------------------------------------------------------------------------

class LikelihoodFunction:
    """Callable class for negative log-likelihood optimization."""
    
    def __init__(self, energy: np.ndarray, obs_counts: np.ndarray, 
                 bin_edges: np.ndarray, physics_params: Dict[str, Any],
                 cfg: Config, scale: float = DEFAULT_SCALE):
        self.energy = energy
        self.obs_counts = obs_counts
        self.bin_edges = bin_edges
        self.physics_params = physics_params
        self.cfg = cfg
        self.scale = scale
        self.model_fn = get_model(cfg.model)
        
        # Pre-compute constant values
        S_eff_const = effective_entropy(cfg.S_scalar, cfg.alpha)
        self.base_params = physics_params | {
            "alpha": cfg.alpha,
            "S_scalar": cfg.S_scalar,
            "S_eff_const": S_eff_const,
        }
    
    def __call__(self, kappa: float, c: float) -> float:
        """Return negative log-likelihood for minimization."""
        params = self.base_params | {"kappa": kappa, "c": c}
        pred = self.model_fn(self.energy, params)
        
        # Bin predicted spectrum
        pred_counts, _ = np.histogram(self.energy, bins=self.bin_edges, 
                                    weights=pred * self.scale)
        pred_counts = pred_counts.astype(float)
        
        # Return negative log-likelihood for minimization
        return -log_poisson_likelihood(self.obs_counts, pred_counts)


# -----------------------------------------------------------------------------
# iminuit optimization
# -----------------------------------------------------------------------------

def fit_with_iminuit(likelihood_fn: LikelihoodFunction, cfg: Config) -> Dict[str, Any]:
    """Perform continuous optimization using iminuit."""
    try:
        from iminuit import Minuit
    except ImportError:
        raise ImportError("iminuit not available. Install with: pip install iminuit")
    
    # Initial parameter values (center of grid)
    kappa_init = (cfg.kappa_min + cfg.kappa_max) / 2
    c_init = (cfg.c_min + cfg.c_max) / 2
    
    # Create Minuit instance
    m = Minuit(likelihood_fn, kappa=kappa_init, c=c_init)
    
    # Set parameter limits
    m.limits["kappa"] = (cfg.kappa_min, cfg.kappa_max)
    m.limits["c"] = (cfg.c_min, cfg.c_max)
    
    # Set reasonable errors for initial step sizes
    m.errors["kappa"] = (cfg.kappa_max - cfg.kappa_min) / 100
    m.errors["c"] = (cfg.c_max - cfg.c_min) / 100
    
    # Perform minimization
    m.migrad()
    
    # Extract results
    result = {
        "method": "iminuit",
        "success": m.valid,
        "kappa": m.values["kappa"],
        "c": m.values["c"],
        "kappa_err": m.errors["kappa"] if m.valid else np.nan,
        "c_err": m.errors["c"] if m.valid else np.nan,
        "neg_logL": m.fval,
        "logL": -m.fval,
        "nfev": m.nfcn,
        "hesse_valid": m.accurate if hasattr(m, 'accurate') else False,
    }
    
    # Add covariance matrix if available
    if m.valid and hasattr(m, 'covariance'):
        try:
            cov = m.covariance
            result["cov_kappa_kappa"] = cov[0, 0]
            result["cov_c_c"] = cov[1, 1]
            result["cov_kappa_c"] = cov[0, 1]
        except:
            pass
    
    return result


# -----------------------------------------------------------------------------
# Fallback coarse grid optimization
# -----------------------------------------------------------------------------

def fit_with_grid(likelihood_fn: LikelihoodFunction, cfg: Config) -> Dict[str, Any]:
    """Fallback coarse grid search for optimization."""
    print("Using fallback coarse grid optimization...")
    
    # Use fewer steps for coarse grid
    kappa_steps = min(cfg.kappa_steps, 20)
    c_steps = min(cfg.c_steps, 15)
    
    kappas = np.linspace(cfg.kappa_min, cfg.kappa_max, kappa_steps)
    cs = np.linspace(cfg.c_min, cfg.c_max, c_steps)
    
    best_logL = -np.inf
    best_kappa = kappas[kappa_steps // 2]
    best_c = cs[c_steps // 2]
    nfev = 0
    
    for kappa in kappas:
        for c in cs:
            neg_logL = likelihood_fn(kappa, c)
            logL = -neg_logL
            nfev += 1
            
            if logL > best_logL:
                best_logL = logL
                best_kappa = kappa
                best_c = c
    
    return {
        "method": "grid",
        "success": True,
        "kappa": best_kappa,
        "c": best_c,
        "kappa_err": np.nan,
        "c_err": np.nan,
        "neg_logL": -best_logL,
        "logL": best_logL,
        "nfev": nfev,
        "hesse_valid": False,
    }


# -----------------------------------------------------------------------------
# MCMC sampling with emcee
# -----------------------------------------------------------------------------

def sample_with_emcee(likelihood_fn: LikelihoodFunction, cfg: Config, 
                     best_fit: Dict[str, Any], n_walkers: int = 32, 
                     n_steps: int = 1000, burn_in: int = 200) -> Dict[str, Any]:
    """Perform MCMC sampling using emcee."""
    try:
        import emcee
    except ImportError:
        raise ImportError("emcee not available. Install with: pip install emcee")
    
    def log_prior(theta):
        """Uniform prior within parameter bounds."""
        kappa, c = theta
        if cfg.kappa_min <= kappa <= cfg.kappa_max and cfg.c_min <= c <= cfg.c_max:
            return 0.0
        return -np.inf
    
    def log_probability(theta):
        """Log posterior = log prior + log likelihood."""
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - likelihood_fn(*theta)  # likelihood_fn returns negative logL
    
    # Initialize walkers around best fit
    ndim = 2
    kappa_spread = (cfg.kappa_max - cfg.kappa_min) / 20
    c_spread = (cfg.c_max - cfg.c_min) / 20
    
    pos = []
    for _ in range(n_walkers):
        kappa = best_fit["kappa"] + np.random.normal(0, kappa_spread)
        c = best_fit["c"] + np.random.normal(0, c_spread)
        # Ensure within bounds
        kappa = np.clip(kappa, cfg.kappa_min, cfg.kappa_max)
        c = np.clip(c, cfg.c_min, cfg.c_max)
        pos.append([kappa, c])
    
    # Create sampler and run
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability)
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Extract samples after burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)
    
    # Compute statistics
    kappa_samples = samples[:, 0]
    c_samples = samples[:, 1]
    
    result = {
        "method": "emcee",
        "n_samples": len(samples),
        "kappa_mean": np.mean(kappa_samples),
        "kappa_std": np.std(kappa_samples),
        "kappa_median": np.median(kappa_samples),
        "kappa_16": np.percentile(kappa_samples, 16),
        "kappa_84": np.percentile(kappa_samples, 84),
        "c_mean": np.mean(c_samples),
        "c_std": np.std(c_samples),
        "c_median": np.median(c_samples),
        "c_16": np.percentile(c_samples, 16),
        "c_84": np.percentile(c_samples, 84),
        "acceptance_fraction": np.mean(sampler.acceptance_fraction),
    }
    
    # Add covariance
    cov = np.cov(samples.T)
    result["cov_kappa_kappa"] = cov[0, 0]
    result["cov_c_c"] = cov[1, 1]
    result["cov_kappa_c"] = cov[0, 1]
    
    return result, samples


# -----------------------------------------------------------------------------
# Main fitting function
# -----------------------------------------------------------------------------

def fit_single_file(filepath: Path, cfg: Config, use_mcmc: bool = False,
                   mcmc_params: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Fit a single data file with continuous optimization."""
    
    # Load and validate data
    try:
        df = auto_detect_and_parse(filepath)
    except Exception as e:
        raise ValueError(f"Failed to load data from {filepath}: {e}")
    
    if df.empty:
        raise ValueError(f"Data file {filepath} is empty")
    
    energy = df["Energy"].to_numpy()
    observed = df["Obs"].to_numpy()
    
    # Validate data
    if len(energy) != len(observed) or len(energy) == 0:
        raise ValueError(f"Invalid data dimensions in {filepath}")
    
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        raise ValueError(f"Invalid energy values in {filepath}")
    
    if not np.all(np.isfinite(observed)):
        raise ValueError(f"Invalid observed values in {filepath}")
    
    print(f"\nFitting {filepath.name}:")
    print(f"  Data points: {len(df)}")
    print(f"  Energy range: {energy.min():.2f} to {energy.max():.2f}")
    print(f"  Observable range: {observed.min():.2e} to {observed.max():.2e}")
    
    # Prepare physics parameters
    physics_params = {
        "m_H": DEFAULT_HIGGS_MASS,
        "Gamma": DEFAULT_HIGGS_WIDTH,
        "N": DEFAULT_NORMALIZATION,
        "entropy_shape": cfg.entropy_shape,
    }
    
    # Bin the data
    n_bins = DEFAULT_BINS
    scale = DEFAULT_SCALE
    bin_edges, obs_counts, _ = make_counts(energy, observed, n_bins=n_bins, scale=scale)
    
    # Create likelihood function
    likelihood_fn = LikelihoodFunction(energy, obs_counts, bin_edges, 
                                     physics_params, cfg, scale)
    
    # Try iminuit first, fallback to grid
    try:
        result = fit_with_iminuit(likelihood_fn, cfg)
        print(f"  iminuit optimization: kappa={result['kappa']:.4f}±{result['kappa_err']:.4f}, "
              f"c={result['c']:.4f}±{result['c_err']:.4f}, logL={result['logL']:.3f}")
    except ImportError:
        result = fit_with_grid(likelihood_fn, cfg)
        print(f"  Grid optimization: kappa={result['kappa']:.4f}, "
              f"c={result['c']:.4f}, logL={result['logL']:.3f}")
    
    # Add file metadata
    result["filename"] = filepath.name
    result["data_points"] = len(energy)
    result["energy_min"] = energy.min()
    result["energy_max"] = energy.max()
    result["obs_min"] = observed.min()
    result["obs_max"] = observed.max()
    result["entropy_shape"] = cfg.entropy_shape
    
    # Add data file metadata if available
    metadata = getattr(df, 'attrs', {}) if hasattr(df, 'attrs') else {}
    if metadata:
        for key, value in metadata.items():
            result[f"meta_{key}"] = value
    
    # Optional MCMC sampling
    if use_mcmc:
        try:
            mcmc_defaults = {"n_walkers": 32, "n_steps": 1000, "burn_in": 200}
            if mcmc_params:
                mcmc_defaults.update(mcmc_params)
            
            print("  Running MCMC sampling...")
            mcmc_result, samples = sample_with_emcee(likelihood_fn, cfg, result, **mcmc_defaults)
            result["mcmc"] = mcmc_result
            print(f"  MCMC results: kappa={mcmc_result['kappa_median']:.4f}"
                  f"+{mcmc_result['kappa_84']-mcmc_result['kappa_median']:.4f}"
                  f"-{mcmc_result['kappa_median']-mcmc_result['kappa_16']:.4f}, "
                  f"c={mcmc_result['c_median']:.4f}"
                  f"+{mcmc_result['c_84']-mcmc_result['c_median']:.4f}"
                  f"-{mcmc_result['c_median']-mcmc_result['c_16']:.4f}")
            
        except ImportError:
            print("  emcee not available, skipping MCMC")
            result["mcmc"] = None
    
    return result


# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def main(cfg: Config) -> None:
    """Main CLI interface for continuous fitting."""
    
    # Only process single files for continuous fitting
    if cfg.batch_mode:
        raise ValueError("Continuous fitting only supports single files. "
                        "Use sweeps.py for batch processing.")
    
    if cfg.calibration:
        filepath = Path("toy.csv")
    else:
        filepath = cfg.data_path
    
    if not filepath.exists():
        raise ValueError(f"Data file not found: {filepath}")
    
    # Set up output directory
    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Perform fitting
    print("Starting continuous optimization...")
    result = fit_single_file(filepath, cfg, use_mcmc=False)  # MCMC can be added later
    
    # Save results
    fit_file = out_dir / "fit_results.csv"
    
    # Load existing results or create new
    if fit_file.exists():
        try:
            existing_df = pd.read_csv(fit_file)
            # Remove existing entry for this file
            existing_df = existing_df[existing_df['filename'] != result['filename']]
            # Add new result
            new_df = pd.DataFrame([result])
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not read existing fit results: {e}")
            combined_df = pd.DataFrame([result])
    else:
        combined_df = pd.DataFrame([result])
    
    # Sort and save
    combined_df = combined_df.sort_values('filename')
    combined_df.to_csv(fit_file, index=False)
    
    print(f"\nFit results saved to {fit_file}")
    print(f"Best fit: kappa={result['kappa']:.6f}, c={result['c']:.6f}, logL={result['logL']:.6f}")


if __name__ == "__main__":
    from config import parse_args
    main(parse_args())