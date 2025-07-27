# version 0.1
"""
Unit smoke tests for the continuous fitting module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fit import fit_single_file, LikelihoodFunction
from config import Config
from constants import DEFAULT_BINS, DEFAULT_SCALE, DEFAULT_HIGGS_MASS


def test_toy_calibration():
    """Smoke test: ensure iminuit converges on toy.csv data."""
    # Create config for calibration
    cfg = Config(
        calibration=True,
        kappa_min=0.5,
        kappa_max=1.5,
        kappa_steps=20,
        c_min=0.0,
        c_max=2.0,
        c_steps=15,
        alpha=0.04,
        S_scalar=0.10,
        entropy_shape="log",
        output=Path("test_results")
    )
    cfg.validate()
    
    # Run fit
    result = fit_single_file(Path("toy.csv"), cfg, use_mcmc=False)
    
    # Basic sanity checks
    assert result["method"] == "iminuit"
    assert result["success"] is True
    assert 0.5 <= result["kappa"] <= 1.5
    assert 0.0 <= result["c"] <= 2.0
    assert result["logL"] > -2000  # Should be reasonable
    assert result["nfev"] > 0
    assert np.isfinite(result["kappa_err"])
    assert np.isfinite(result["c_err"])


def test_likelihood_function():
    """Test that LikelihoodFunction works correctly."""
    # Create synthetic data
    energy = np.linspace(120, 130, 50)
    observed = np.exp(-0.5 * ((energy - 125) / 2) ** 2) + 0.01
    
    # Create config
    cfg = Config(
        kappa_min=0.5,
        kappa_max=1.5,
        c_min=0.0,
        c_max=1.0,
        alpha=0.04,
        S_scalar=0.10,
        entropy_shape="log"
    )
    
    # Physics parameters
    physics_params = {
        "m_H": DEFAULT_HIGGS_MASS,
        "Gamma": 0.004,
        "N": 1.0,
        "entropy_shape": "log",
    }
    
    # Bin the data
    from sweeps import make_counts
    bin_edges, obs_counts, _ = make_counts(energy, observed, 
                                          n_bins=DEFAULT_BINS, 
                                          scale=DEFAULT_SCALE)
    
    # Create likelihood function
    likelihood_fn = LikelihoodFunction(energy, obs_counts, bin_edges, 
                                     physics_params, cfg, DEFAULT_SCALE)
    
    # Test that it returns finite values
    neg_logL1 = likelihood_fn(1.0, 0.5)
    neg_logL2 = likelihood_fn(1.1, 0.6)
    
    assert np.isfinite(neg_logL1)
    assert np.isfinite(neg_logL2)
    assert neg_logL1 != neg_logL2  # Should be different for different parameters


def test_iminuit_available():
    """Test that iminuit is available and working."""
    try:
        from iminuit import Minuit
        # Simple test function
        def simple_chi2(x, y):
            return (x - 1) ** 2 + (y - 2) ** 2
        
        m = Minuit(simple_chi2, x=0, y=0)
        m.migrad()
        
        assert m.valid
        assert abs(m.values["x"] - 1.0) < 0.01
        assert abs(m.values["y"] - 2.0) < 0.01
        
    except ImportError:
        pytest.skip("iminuit not available")


if __name__ == "__main__":
    # Run tests directly
    test_iminuit_available()
    test_likelihood_function()
    test_toy_calibration()
    print("All tests passed!")