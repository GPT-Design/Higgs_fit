"""
Data loader module for Higgs entropy fitting.

Supports multiple file formats:
- CSV files (Energy, Obs columns)
- Parquet files (Energy, Obs columns)
- HDF5 files (LIGO GWOSC format)

For LIGO HDF5 files, converts gravitational wave strain data to
spectral power density suitable for entropy fitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import h5py
from scipy import signal
from scipy.fft import fft, fftfreq


class DataLoader:
    """Unified data loader for various file formats."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing data files (default: ./data)
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        
    def list_files(self, pattern: str = "*") -> List[Path]:
        """List all files matching pattern in data directory."""
        if not self.data_dir.exists():
            return []
        return list(self.data_dir.glob(pattern))
    
    def load_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from file with automatic format detection.
        
        Args:
            filepath: Path to data file
            
        Returns:
            DataFrame with columns ['Energy', 'Obs']
            
        Raises:
            ValueError: If file format is not supported
        """
        filepath = Path(filepath)
        
        # Make path relative to data directory if it's just a filename
        if not filepath.is_absolute() and not filepath.exists():
            filepath = self.data_dir / filepath
        
        if not filepath.exists():
            raise ValueError(f"File not found: {filepath}")
            
        suffix = filepath.suffix.lower()
        
        if suffix == ".csv":
            return self._load_csv(filepath)
        elif suffix in [".parquet", ".pq"]:
            return self._load_parquet(filepath)
        elif suffix in [".hdf5", ".h5"]:
            return self._load_hdf5(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_csv(self, filepath: Path) -> pd.DataFrame:
        """Load CSV file with Energy, Obs columns."""
        df = pd.read_csv(filepath)
        return self._validate_columns(df, filepath)
    
    def _load_parquet(self, filepath: Path) -> pd.DataFrame:
        """Load Parquet file with Energy, Obs columns."""
        df = pd.read_parquet(filepath)
        return self._validate_columns(df, filepath)
    
    def _load_hdf5(self, filepath: Path) -> pd.DataFrame:
        """
        Load LIGO HDF5 file and convert to Energy/Obs format.
        
        For LIGO files, converts strain time series to frequency domain
        power spectral density, interpreting frequency as "Energy" and
        PSD as "Obs".
        """
        with h5py.File(filepath, 'r') as f:
            # Extract strain data
            strain = f['strain/Strain'][:]
            
            # Extract metadata
            duration = f['meta/Duration'][()]
            sample_rate = len(strain) / duration
            detector = f['meta/Detector'][()].decode()
            
            # Validate strain data
            if len(strain) == 0:
                raise ValueError(f"Empty strain data in {filepath}")
            
            if not np.all(np.isfinite(strain)):
                raise ValueError(f"Non-finite strain data in {filepath}")
        
        # Convert to frequency domain power spectral density
        # Using Welch's method for robust PSD estimation
        frequencies, psd = signal.welch(
            strain, 
            fs=sample_rate,
            window='hann',
            nperseg=len(strain)//8,  # Use 1/8 of data for each segment
            noverlap=None,
            detrend='constant',
            scaling='density'
        )
        
        # Filter to reasonable frequency range (avoid DC and Nyquist)
        valid_mask = (frequencies > 10) & (frequencies < sample_rate/2 - 100)
        frequencies = frequencies[valid_mask]
        psd = psd[valid_mask]
        
        # Create DataFrame
        df = pd.DataFrame({
            'Energy': frequencies,  # Frequency as proxy for "Energy"
            'Obs': psd             # Power spectral density as "Observable"
        })
        
        # Add metadata as attributes
        df.attrs['detector'] = detector
        df.attrs['duration'] = duration
        df.attrs['sample_rate'] = sample_rate
        df.attrs['filepath'] = str(filepath)
        
        return df
    
    def _validate_columns(self, df: pd.DataFrame, filepath: Path) -> pd.DataFrame:
        """Validate that DataFrame has required Energy and Obs columns."""
        required_columns = ['Energy', 'Obs']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Missing columns {missing_columns} in {filepath}. "
                f"Found columns: {list(df.columns)}"
            )
        
        return df
    
    def load_multiple_files(self, pattern: str = "*.hdf5") -> Dict[str, pd.DataFrame]:
        """
        Load multiple files matching pattern.
        
        Args:
            pattern: Glob pattern for files to load
            
        Returns:
            Dictionary mapping filename to DataFrame
        """
        files = self.list_files(pattern)
        results = {}
        
        for filepath in files:
            try:
                df = self.load_file(filepath)
                results[filepath.name] = df
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue
        
        return results
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict:
        """
        Get metadata information about a file without loading full data.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        # Make path relative to data directory if needed
        if not filepath.is_absolute() and not filepath.exists():
            filepath = self.data_dir / filepath
        
        if not filepath.exists():
            raise ValueError(f"File not found: {filepath}")
        
        info = {
            'filepath': str(filepath),
            'filename': filepath.name,
            'size_bytes': filepath.stat().st_size,
            'format': filepath.suffix.lower(),
        }
        
        # Add format-specific info
        if filepath.suffix.lower() in ['.hdf5', '.h5']:
            with h5py.File(filepath, 'r') as f:
                info.update({
                    'detector': f['meta/Detector'][()].decode(),
                    'duration': f['meta/Duration'][()],
                    'gps_start': f['meta/GPSstart'][()],
                    'description': f['meta/Description'][()].decode(),
                    'strain_samples': f['strain/Strain'].shape[0],
                })
        
        return info


def create_synthetic_energy_data(
    frequencies: np.ndarray, 
    psd: np.ndarray,
    energy_range: Tuple[float, float] = (100.0, 150.0),
    n_points: int = 400
) -> pd.DataFrame:
    """
    Convert frequency/PSD data to synthetic energy/observable data.
    
    This function maps frequency domain data to energy domain for
    compatibility with Higgs entropy fitting.
    
    Args:
        frequencies: Frequency array (Hz)
        psd: Power spectral density array
        energy_range: Energy range in GeV to map to
        n_points: Number of energy points to generate
        
    Returns:
        DataFrame with Energy (GeV) and Obs columns
    """
    # Create energy grid
    energy = np.linspace(energy_range[0], energy_range[1], n_points)
    
    # Map frequencies to energy range
    freq_min, freq_max = frequencies.min(), frequencies.max()
    mapped_freqs = freq_min + (freq_max - freq_min) * (energy - energy_range[0]) / (energy_range[1] - energy_range[0])
    
    # Interpolate PSD values to energy grid
    observable = np.interp(mapped_freqs, frequencies, psd)
    
    # Add some smoothing to make it more realistic
    from scipy.ndimage import gaussian_filter1d
    observable = gaussian_filter1d(observable, sigma=2.0)
    
    # Normalize to reasonable range
    observable = (observable - observable.min()) / (observable.max() - observable.min())
    observable = 0.001 + 0.004 * observable  # Scale to [0.001, 0.005] range
    
    return pd.DataFrame({
        'Energy': energy,
        'Obs': observable
    })


# Convenience functions for common operations
def load_ligo_file(filepath: Union[str, Path]) -> pd.DataFrame:
    """Convenience function to load a single LIGO HDF5 file."""
    loader = DataLoader()
    return loader.load_file(filepath)


def list_ligo_files(data_dir: Optional[Union[str, Path]] = None) -> List[Path]:
    """Convenience function to list all LIGO HDF5 files."""
    loader = DataLoader(data_dir)
    return loader.list_files("*.hdf5")


def load_all_ligo_files(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, pd.DataFrame]:
    """Convenience function to load all LIGO HDF5 files."""
    loader = DataLoader(data_dir)
    return loader.load_multiple_files("*.hdf5")