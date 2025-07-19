"""
File parser module with auto-detection capabilities.

Provides high-level interface for parsing various data formats
and converting them to the standard Energy/Obs format used by
the Higgs entropy fitting pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from data_loader import DataLoader, create_synthetic_energy_data


class FileParser:
    """High-level file parser with auto-detection and conversion."""
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize file parser.
        
        Args:
            data_dir: Directory containing data files (default: ./data)
        """
        self.loader = DataLoader(data_dir)
        self.data_dir = self.loader.data_dir
    
    def parse_file(self, filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Parse file with automatic format detection and conversion.
        
        Args:
            filepath: Path to file
            **kwargs: Additional arguments for conversion
            
        Returns:
            DataFrame with Energy and Obs columns suitable for fitting
        """
        filepath = Path(filepath)
        
        # Load raw data
        df = self.loader.load_file(filepath)
        
        # Convert based on file type and content
        if filepath.suffix.lower() in ['.hdf5', '.h5']:
            # For LIGO files, convert frequency/PSD to energy/observable
            return self._convert_ligo_data(df, **kwargs)
        else:
            # For CSV/Parquet, assume already in Energy/Obs format
            return self._validate_energy_obs_data(df)
    
    def _convert_ligo_data(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Convert LIGO frequency/PSD data to Energy/Obs format."""
        # Extract conversion parameters
        energy_range = kwargs.get('energy_range', (100.0, 150.0))
        n_points = kwargs.get('n_points', 400)
        
        # Convert using synthetic energy mapping
        converted_df = create_synthetic_energy_data(
            df['Energy'].values,  # Frequencies
            df['Obs'].values,     # PSD values
            energy_range=energy_range,
            n_points=n_points
        )
        
        # Preserve metadata
        converted_df.attrs.update(df.attrs)
        
        return converted_df
    
    def _validate_energy_obs_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean Energy/Obs data."""
        # Remove any rows with non-finite values
        df = df[df['Energy'].notna() & df['Obs'].notna()]
        df = df[np.isfinite(df['Energy']) & np.isfinite(df['Obs'])]
        
        # Sort by energy
        df = df.sort_values('Energy').reset_index(drop=True)
        
        # Ensure energy values are positive
        df = df[df['Energy'] > 0]
        
        return df
    
    def list_available_files(self) -> Dict[str, List[Path]]:
        """List all available files by type."""
        files = {
            'csv': self.loader.list_files("*.csv"),
            'parquet': self.loader.list_files("*.parquet") + self.loader.list_files("*.pq"),
            'hdf5': self.loader.list_files("*.hdf5") + self.loader.list_files("*.h5"),
        }
        return files
    
    def get_file_summary(self, filepath: Union[str, Path]) -> Dict:
        """Get comprehensive file information."""
        info = self.loader.get_file_info(filepath)
        
        # Add parsing-specific information
        try:
            df = self.parse_file(filepath)
            info.update({
                'data_points': len(df),
                'energy_range': (df['Energy'].min(), df['Energy'].max()),
                'obs_range': (df['Obs'].min(), df['Obs'].max()),
                'energy_units': 'GeV' if Path(filepath).suffix.lower() in ['.hdf5', '.h5'] else 'unknown',
                'parseable': True
            })
        except Exception as e:
            info.update({
                'parseable': False,
                'parse_error': str(e)
            })
        
        return info
    
    def find_best_files(self, 
                       min_points: int = 100,
                       energy_range: Optional[tuple] = None) -> List[Path]:
        """
        Find files that meet criteria for fitting.
        
        Args:
            min_points: Minimum number of data points
            energy_range: Required energy range (min, max)
            
        Returns:
            List of suitable file paths
        """
        all_files = []
        for file_list in self.list_available_files().values():
            all_files.extend(file_list)
        
        suitable_files = []
        
        for filepath in all_files:
            try:
                summary = self.get_file_summary(filepath)
                
                # Check if parseable
                if not summary.get('parseable', False):
                    continue
                
                # Check minimum points
                if summary['data_points'] < min_points:
                    continue
                
                # Check energy range if specified
                if energy_range:
                    file_range = summary['energy_range']
                    if file_range[0] > energy_range[0] or file_range[1] < energy_range[1]:
                        continue
                
                suitable_files.append(filepath)
                
            except Exception:
                continue
        
        return suitable_files


def auto_detect_and_parse(filepath: Union[str, Path], 
                         data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Convenience function for automatic file detection and parsing.
    
    Args:
        filepath: Path to file
        data_dir: Data directory (default: ./data)
        
    Returns:
        DataFrame ready for entropy fitting
    """
    parser = FileParser(data_dir)
    return parser.parse_file(filepath)


def list_parseable_files(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, List[Path]]:
    """
    Convenience function to list all parseable files.
    
    Args:
        data_dir: Data directory (default: ./data)
        
    Returns:
        Dictionary of file types and paths
    """
    parser = FileParser(data_dir)
    return parser.list_available_files()


def get_data_summary(data_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Get summary of all available data files.
    
    Args:
        data_dir: Data directory (default: ./data)
        
    Returns:
        DataFrame with file summaries
    """
    parser = FileParser(data_dir)
    all_files = []
    for file_list in parser.list_available_files().values():
        all_files.extend(file_list)
    
    summaries = []
    for filepath in all_files:
        try:
            summary = parser.get_file_summary(filepath)
            summaries.append(summary)
        except Exception as e:
            summaries.append({
                'filepath': str(filepath),
                'filename': filepath.name,
                'parseable': False,
                'error': str(e)
            })
    
    return pd.DataFrame(summaries)


