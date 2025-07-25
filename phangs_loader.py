# version 0.1
"""
PHANGS-ALMA CO (2-1) cube loader for shock front extraction.

Reads FITS or CASA image cubes and extracts shock front properties
using velocity dispersion thresholds or external masks.
"""

from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union
import numpy as np

try:
    from astropy.io import fits
    from astropy.wcs import WCS
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def load_co_cube(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load CO (2-1) cube from FITS file.
    
    Args:
        filepath: Path to FITS cube file
        
    Returns:
        data: 3D array [velocity, dec, ra] in K
        header: Dictionary with WCS and metadata
        
    Raises:
        ImportError: If astropy not available
        ValueError: If file format unsupported
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy required for FITS I/O. Install with: pip install astropy")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise ValueError(f"File not found: {filepath}")
    
    # Load FITS file
    with fits.open(filepath) as hdul:
        data = hdul[0].data
        header = hdul[0].header
    
    # Validate data shape (expect 3D or 4D)
    if data.ndim == 4:
        # Remove Stokes axis if present (first axis)
        data = data[0]
        print(f"Removed Stokes axis, new shape: {data.shape}")
    elif data.ndim != 3:
        raise ValueError(f"Expected 3D or 4D data, got {data.ndim}D")
    
    # Ensure data is finite
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Extract WCS and metadata
    wcs = WCS(header)
    metadata = {
        'wcs': wcs,
        'shape': data.shape,
        'object': header.get('OBJECT', 'unknown'),
        'bunit': header.get('BUNIT', 'K'),
        'bmaj': header.get('BMAJ', np.nan),  # beam major axis
        'bmin': header.get('BMIN', np.nan),  # beam minor axis
        'cdelt3': header.get('CDELT3', 1.0),  # velocity resolution
    }
    
    return data, metadata


def compute_velocity_dispersion(data: np.ndarray, velocity_axis: np.ndarray) -> np.ndarray:
    """
    Compute velocity dispersion map from CO cube.
    
    Args:
        data: 3D CO intensity cube [velocity, dec, ra]
        velocity_axis: Velocity coordinates in km/s
        
    Returns:
        sigma_v: 2D velocity dispersion map in km/s
    """
    nv, ny, nx = data.shape
    sigma_v = np.zeros((ny, nx))
    
    for j in range(ny):
        for i in range(nx):
            spectrum = data[:, j, i]
            
            # Skip if no significant emission
            if np.max(spectrum) < 3 * np.std(spectrum):
                continue
            
            # Compute intensity-weighted velocity dispersion
            weights = np.maximum(spectrum, 0)  # No negative weights
            if np.sum(weights) == 0:
                continue
                
            v_mean = np.average(velocity_axis, weights=weights)
            v_var = np.average((velocity_axis - v_mean)**2, weights=weights)
            sigma_v[j, i] = np.sqrt(v_var)
    
    return sigma_v


def extract_shock_fronts(
    data: np.ndarray,
    velocity_axis: np.ndarray,
    threshold: float = 15.0,
    mask: Optional[np.ndarray] = None,
    min_pixels: int = 5
) -> List[Dict]:
    """
    Extract shock front properties from CO cube.
    
    Args:
        data: 3D CO intensity cube [velocity, dec, ra] in K
        velocity_axis: Velocity coordinates in km/s
        threshold: Velocity dispersion threshold in km/s
        mask: Optional 2D boolean mask for shock regions
        min_pixels: Minimum pixels per shock region
        
    Returns:
        shocks: List of shock dictionaries with properties
    """
    # Compute velocity dispersion
    sigma_v_map = compute_velocity_dispersion(data, velocity_axis)
    
    # Compute integrated intensity
    I_CO_map = np.trapz(data, velocity_axis, axis=0)
    
    # Apply threshold or mask
    if mask is not None:
        shock_mask = mask.astype(bool)
    else:
        shock_mask = sigma_v_map > threshold
    
    # Label connected regions (simple approach)
    shock_regions = label_regions(shock_mask, min_pixels=min_pixels)
    
    shocks = []
    for region_id in np.unique(shock_regions):
        if region_id == 0:  # Skip background
            continue
            
        # Extract region properties
        region_mask = shock_regions == region_id
        region_pixels = np.sum(region_mask)
        
        if region_pixels < min_pixels:
            continue
        
        # Mean properties over region
        sigma_v_mean = np.mean(sigma_v_map[region_mask])
        I_CO_mean = np.mean(I_CO_map[region_mask])
        
        # Estimate upstream density (placeholder - would need additional data)
        # For now, use typical ISM density
        rho_upstream = 100.0  # cm^-3
        
        # Estimate shock velocity from dispersion (crude approximation)
        v_shock = sigma_v_mean * 2.0  # km/s, factor from shock theory
        
        shock = {
            'upstream_density': rho_upstream,  # cm^-3
            'shock_velocity': v_shock,         # km/s
            'sigma_v_obs': sigma_v_mean,       # km/s
            'I_CO_obs': I_CO_mean,            # K km/s
            'region_id': region_id,
            'n_pixels': region_pixels,
        }
        shocks.append(shock)
    
    return shocks


def label_regions(mask: np.ndarray, min_pixels: int = 5) -> np.ndarray:
    """
    Simple connected component labeling for 2D boolean mask.
    
    Args:
        mask: 2D boolean array
        min_pixels: Minimum pixels per region
        
    Returns:
        labels: 2D integer array with region labels
    """
    try:
        from scipy.ndimage import label
        labeled, n_features = label(mask)
        
        # Remove small regions
        for i in range(1, n_features + 1):
            if np.sum(labeled == i) < min_pixels:
                labeled[labeled == i] = 0
        
        return labeled
        
    except ImportError:
        # Fallback: treat entire mask as single region
        print("Warning: scipy not available, using single region")
        return mask.astype(int)


def load_mask(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load 2D boolean mask from FITS file.
    
    Args:
        filepath: Path to mask FITS file
        
    Returns:
        mask: 2D boolean array
    """
    if not ASTROPY_AVAILABLE:
        raise ImportError("astropy required for FITS I/O")
    
    with fits.open(filepath) as hdul:
        data = hdul[0].data
    
    # Handle 4D mask (remove Stokes and velocity axes)
    if data.ndim == 4:
        # Take first Stokes, integrate over velocity
        data = data[0]  # Remove Stokes
        data = np.any(data > 0, axis=0)  # Collapse velocity axis
        print(f"Processed 4D mask to 2D, shape: {data.shape}")
    elif data.ndim == 3:
        # Integrate over velocity
        data = np.any(data > 0, axis=0)
    elif data.ndim != 2:
        raise ValueError(f"Expected 2D, 3D or 4D mask, got {data.ndim}D")
    
    # Convert to boolean
    return data.astype(bool)


def load_phangs_data(
    cube_path: Union[str, Path],
    mask_path: Optional[Union[str, Path]] = None,
    threshold: float = 15.0,
    min_pixels: int = 5,
    noise_file: Optional[Union[str, Path]] = None
) -> Tuple[List[Dict], Dict]:
    """
    Main interface: load PHANGS cube and extract shock properties.
    
    Args:
        cube_path: Path to CO cube FITS file
        mask_path: Optional path to shock mask FITS file
        threshold: Velocity dispersion threshold in km/s (if no mask)
        min_pixels: Minimum pixels per shock region
        noise_file: Optional path to separate noise FITS file
        
    Returns:
        shocks: List of shock front dictionaries
        metadata: Cube metadata and processing info
    """
    print(f"Loading CO cube: {cube_path}")
    data, cube_metadata = load_co_cube(cube_path)
    
    # Create velocity axis from WCS
    nv = data.shape[0]
    wcs = cube_metadata['wcs']
    
    try:
        # Convert frequency to velocity (assuming FREQ axis is 3rd)
        freq_axis = wcs.spectral.pixel_to_world_values(np.arange(nv))
        # Convert Hz to km/s (crude approximation for demo)
        # Proper conversion would need rest frequency
        velocity_axis = (freq_axis - freq_axis[nv//2]) / freq_axis[nv//2] * 3e5  # km/s
        velocity_axis = velocity_axis[::-1]  # Reverse to ascending velocity
        print(f"Velocity range: {velocity_axis.min():.1f} to {velocity_axis.max():.1f} km/s")
    except:
        # Fallback to simple linear axis
        print("Warning: Using fallback velocity axis")
        dv = cube_metadata.get('cdelt3', 1.0)  # km/s  
        velocity_axis = np.arange(nv) * dv - nv * dv / 2
    
    # Load mask if provided
    mask = None
    if mask_path is not None:
        print(f"Loading shock mask: {mask_path}")
        mask = load_mask(mask_path)
    
    # Load noise data if provided
    noise_data = None
    if noise_file is not None:
        print(f"Loading noise data: {noise_file}")
        noise_data, _ = load_co_cube(noise_file)
        print(f"Noise cube shape: {noise_data.shape}")
    
    # Extract shocks
    print(f"Extracting shock fronts (threshold={threshold} km/s)...")
    shocks = extract_shock_fronts(data, velocity_axis, threshold, mask, min_pixels)
    
    print(f"Found {len(shocks)} shock regions")
    
    # Combine metadata
    metadata = cube_metadata.copy()
    metadata.update({
        'threshold': threshold,
        'min_pixels': min_pixels,
        'n_shocks': len(shocks),
        'velocity_axis': velocity_axis,
        'noise_file': str(noise_file) if noise_file else None,
        'has_noise_data': noise_data is not None,
    })
    
    return shocks, metadata