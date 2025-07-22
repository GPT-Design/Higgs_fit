# Higgs Entropy Fitting

A Python package for fitting Higgs boson data with entropy-aware models, supporting 2D parameter sweeps using Poisson log-likelihood statistics.

## Features

- **Multiple Data Formats**: Supports CSV, Parquet, and HDF5 files
- **LIGO Integration**: Automatic conversion of LIGO gravitational wave data to energy/observable format
- **Entropy Models**: Constant, power-law, and log-running entropy profiles
- **Parameter Sweeps**: 2D grid search over κ (entropy coupling) and c (log-running coefficient)
- **Robust Validation**: Comprehensive input validation and error handling
- **Auto-Detection**: Automatic file format detection and parsing

## Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install iminuit emcee  # For continuous optimization

# 2. Run calibration test
python fit.py --calibration

# 3. Grid sweep on toy data
python sweeps.py --calibration

# 4. Batch process all LIGO data
python sweeps.py  # Grid sweep (fast)
python batch_fit.py  # Continuous optimization (precise)
```

## Installation

```bash
pip install -r requirements.txt
pip install iminuit emcee  # Optional: for continuous optimization
```

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- h5py >= 3.1.0
- iminuit >= 2.0 (optional, for continuous optimization)
- emcee >= 3.0 (optional, for MCMC sampling)

## Usage

### Grid Sweeps (Fast)

```bash
# Grid sweep on toy data
python sweeps.py --calibration

# Grid sweep on specific file
python sweeps.py data/H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5

# Batch process all files in data/
python sweeps.py
```

### Continuous Optimization (Precise)

```bash
# Continuous fit on toy data
python fit.py --calibration

# Continuous fit on specific file
python fit.py data/H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5

# Batch continuous fitting (all files)
python batch_fit.py
```

### PHANGS Shock Fitting

```bash
# Fit PHANGS-ALMA CO shock data with 3-parameter model
python phangs_fit.py data/ngc4254_co21.fits --mask data/shock_mask.fits

# Parameters: kappa_E, kappa_S (tensor couplings), log10_M (Mach number)
# Output: JSON results with MINOS errors and significance vs GR
```

### File Management

```bash
# List all available data files
python list_data.py

# Get detailed information about a specific file
python list_data.py "H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5"
```

### Command Line Options

```bash
python cli.py --help
```

Options include:
- `--entropy-shape {constant,powerlaw,log}`: Entropy profile type
- `--alpha ALPHA`: Entropy coupling strength (default: 0.04)
- `--kappa-range MIN MAX`: κ parameter range (default: 0.0 2.0)
- `--c-range MIN MAX`: c parameter range (default: 0.0 1.5)
- `--kappa-steps STEPS`: Number of κ steps (default: 50)
- `--c-steps STEPS`: Number of c steps (default: 25)
- `--output OUTPUT`: Output directory (default: results)

## Data Formats

### CSV/Parquet Files
Should contain columns:
- `Energy`: Energy values (GeV)
- `Obs`: Observable values

### HDF5 Files (LIGO Format)
Automatically converted from LIGO gravitational wave strain data:
- Strain time series → Power spectral density
- Frequency → Energy (mapped to 100-150 GeV range)
- PSD → Observable

## Physics Model

The fitting combines two components:

```
Observable = Breit_Wigner + κ × S_effective
```

Where:
- **Breit-Wigner**: Standard Higgs resonance at 125 GeV
- **κ**: Entropy coupling coefficient (fitted parameter)
- **S_effective**: Entropy contribution based on selected profile

### Entropy Profiles

1. **Constant**: `S_eff = α × S₀`
2. **Power-law**: `S_eff = α × S₀ × (E/m_H)^β`
3. **Log-running**: `S_eff = α × S₀ × [1 + c × log(E/m_H)]`

## File Structure

```
Higgs_fit/
├── cli.py              # Command-line interface
├── config.py           # Configuration management
├── models.py           # Physics models
├── entropy.py          # Entropy profiles
├── stats.py            # Statistical functions
├── sweeps.py           # Grid sweep implementation
├── data_loader.py      # Data loading utilities
├── file_parser.py      # File parsing with auto-detection
├── list_data.py        # Data exploration utility
├── make_toy.py         # Toy data generation
├── data/               # Data directory
│   ├── *.hdf5         # LIGO HDF5 files
│   └── V1/            # Virgo detector data
└── results/           # Output directory
    └── grid_results.csv
```

## Output

### Grid Sweeps
- `results/grid_results.csv`: Detailed parameter grid with log-likelihood values
- `results/summary_results.csv`: Best-fit parameters for each data file

### Continuous Fits
- `results/fit_results.csv`: Precise optimization results with uncertainties
- `results/fit_results_master.csv`: Batch continuous fitting results

### Testing
```bash
# Run unit tests
python -m pytest tests/test_fit.py -v

# Manual test
python tests/test_fit.py
```

## Example Results

For the toy data (generated with κ=0.9, c=1.0):
```
Best: kappa = 0.8980, c = 1.0000 (logL = -61.551)
```

For LIGO data (real gravitational wave data):
```
Best: kappa = 0.2449, c = 0.0000 (logL = -454.513)
```

## Development

The package is structured for easy extension:
- Add new entropy profiles in `entropy.py`
- Add new physics models in `models.py`
- Add new data formats in `data_loader.py`
- Add new statistical tests in `stats.py`