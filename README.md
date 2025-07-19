# Higgs Entropy Fitting

A Python package for fitting Higgs boson data with entropy-aware models, supporting 2D parameter sweeps using Poisson log-likelihood statistics.

## Features

- **Multiple Data Formats**: Supports CSV, Parquet, and HDF5 files
- **LIGO Integration**: Automatic conversion of LIGO gravitational wave data to energy/observable format
- **Entropy Models**: Constant, power-law, and log-running entropy profiles
- **Parameter Sweeps**: 2D grid search over κ (entropy coupling) and c (log-running coefficient)
- **Robust Validation**: Comprehensive input validation and error handling
- **Auto-Detection**: Automatic file format detection and parsing

## Installation

```bash
pip install -r requirements.txt
```

## Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- h5py >= 3.1.0

## Usage

### Basic Usage

```bash
# Run entropy fitting on a data file
python cli.py <data_file> [options]

# Examples:
python cli.py toy.csv --entropy-shape log
python cli.py "H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5" --entropy-shape log
python cli.py "data/V1/V-V1_GWOSC_16KHZ_R1-1245035064-32.hdf5"
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

Results are saved to `results/grid_results.csv` with columns:
- `kappa`: Entropy coupling coefficient
- `c`: Log-running coefficient
- `logL`: Log-likelihood value

The best-fit parameters are printed to console.

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