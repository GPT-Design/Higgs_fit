# version 0.7
"""
Sweep κ and log‑running coefficient c over a 2‑D grid, compute the **Poisson
log‑likelihood** on binned counts, write a results CSV, and report the best fit.

Debug slice: prints logL at κ ≈ 0.9 across the c‑grid to verify sensitivity.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from config import Config
from models import get_model
from entropy import effective_entropy
from stats import log_poisson_likelihood
from file_parser import auto_detect_and_parse

# -----------------------------------------------------------------------------
# Helper: bin spectra into integer‑like counts
# -----------------------------------------------------------------------------

def make_counts(
    energy: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 10,
    scale: float = 200_000.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_edges, counts, bin_centres) after scaling to pseudo‑counts."""
    bin_edges = np.linspace(energy.min(), energy.max(), n_bins + 1)
    counts, _ = np.histogram(energy, bins=bin_edges, weights=values * scale)
    return bin_edges, counts.astype(float), 0.5 * (bin_edges[:-1] + bin_edges[1:])


# -----------------------------------------------------------------------------
# 2‑D sweep core (log‑likelihood)
# -----------------------------------------------------------------------------

def sweep_grid(
    energy: np.ndarray,
    observed: np.ndarray,
    cfg: Config,
    physics_params: Dict[str, Any],
    *,
    n_bins: int = 10,
    scale: float = 200_000.0,
    filename: str = "unknown",
    metadata: Dict[str, Any] = None,
) -> pd.DataFrame:
    """Return DataFrame over κ × c grid with log‑likelihood values."""

    # Pre‑bin observed spectrum once
    bin_edges, obs_counts, _ = make_counts(energy, observed, n_bins=n_bins, scale=scale)

    kappas = np.linspace(cfg.kappa_min, cfg.kappa_max, cfg.kappa_steps)
    cs = np.linspace(cfg.c_min, cfg.c_max, cfg.c_steps)
    model_fn = get_model(cfg.model)

    # Pre-compute constant values
    S_eff_const = effective_entropy(cfg.S_scalar, cfg.alpha)
    base_params = physics_params | {
        "alpha": cfg.alpha,
        "S_scalar": cfg.S_scalar,
        "S_eff_const": S_eff_const,
    }

    rows: list[dict[str, float]] = []
    for kappa in kappas:
        for c in cs:
            params = base_params | {"kappa": kappa, "c": c}
            pred = model_fn(energy, params)
            
            # Use pre-computed bin edges for efficiency
            pred_counts, _ = np.histogram(energy, bins=bin_edges, weights=pred * scale)
            pred_counts = pred_counts.astype(float)

            logL = log_poisson_likelihood(obs_counts, pred_counts)
            row = {"filename": filename, "kappa": kappa, "c": c, "logL": logL}
            
            # Add metadata if available
            if metadata:
                for key, value in metadata.items():
                    if key not in row:  # Don't overwrite existing columns
                        row[key] = value
            
            rows.append(row)

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# CLI wrapper – load data → run grid sweep → save CSV
# -----------------------------------------------------------------------------

def process_single_file(filepath: Path, cfg: Config, all_grid_results: list, 
                       out_dir: Path, is_first_file: bool = False) -> None:
    """Process a single file and append results to grid and summary."""
    try:
        df = auto_detect_and_parse(filepath)
    except Exception as e:
        print(f"Failed to load data from {filepath}: {e}")
        return
    
    # Validate data is not empty
    if df.empty:
        print(f"Data file {filepath} is empty, skipping")
        return
    
    energy = df["Energy"].to_numpy()
    observed = df["Obs"].to_numpy()
    
    # Validate arrays have same length and contain valid data
    if len(energy) != len(observed):
        print(f"Energy and Obs arrays in {filepath} have different lengths, skipping")
        return
    
    if len(energy) == 0:
        print(f"Energy and Obs arrays in {filepath} are empty, skipping")
        return
    
    # Validate energy values are positive and finite
    if not np.all(np.isfinite(energy)) or np.any(energy <= 0):
        print(f"Invalid energy values in {filepath}, skipping")
        return
    
    # Validate observed values are finite
    if not np.all(np.isfinite(observed)):
        print(f"Invalid observed values in {filepath}, skipping")
        return
    
    # Print data information
    print(f"\nProcessing {filepath.name}:")
    print(f"  Loaded {len(df)} data points")
    print(f"  Energy range: {energy.min():.2f} to {energy.max():.2f}")
    print(f"  Observable range: {observed.min():.2e} to {observed.max():.2e}")
    if hasattr(df, 'attrs') and df.attrs:
        print(f"  Metadata: {df.attrs}")

    physics_params: Dict[str, Any] = {
        "m_H": 125.0,
        "Gamma": 0.004,
        "N": 1.0,
        "entropy_shape": cfg.entropy_shape,
    }

    # Extract filename and metadata for results
    filename = filepath.name
    metadata = getattr(df, 'attrs', {}) if hasattr(df, 'attrs') else {}

    results = sweep_grid(energy, observed, cfg, physics_params, 
                        filename=filename, metadata=metadata)

    # Add results to all grid results
    all_grid_results.append(results)
    
    # Find best fit
    best = results.loc[results["logL"].idxmax()]
    print(f"  Best: kappa = {best['kappa']:.4f}, c = {best['c']:.4f} (logL = {best['logL']:.3f})")
    
    # Update summary results
    summary_file = out_dir / "summary_results.csv"
    summary_row = {
        "filename": filename,
        "best_kappa": best['kappa'],
        "best_c": best['c'],
        "best_logL": best['logL'],
        "entropy_shape": cfg.entropy_shape,
        "data_points": len(energy),
        "energy_min": energy.min(),
        "energy_max": energy.max(),
        "obs_min": observed.min(),
        "obs_max": observed.max(),
    }
    
    # Add metadata to summary
    if metadata:
        for key, value in metadata.items():
            if key not in summary_row:
                summary_row[f"meta_{key}"] = value
    
    # Load existing summary or create new one
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        # Remove existing entry for this file if it exists
        summary_df = summary_df[summary_df['filename'] != filename]
        # Add new entry
        new_row_df = pd.DataFrame([summary_row])
        summary_df = pd.concat([summary_df, new_row_df], ignore_index=True)
    else:
        summary_df = pd.DataFrame([summary_row])
    
    # Sort by filename for easy reading
    summary_df = summary_df.sort_values('filename')
    summary_df.to_csv(summary_file, index=False)


def main(cfg: Config) -> None:
    out_dir = Path(cfg.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_grid_results = []
    
    if cfg.batch_mode:
        # Batch process all files in data directory
        from file_parser import list_parseable_files
        files_by_type = list_parseable_files()
        
        all_files = []
        for file_list in files_by_type.values():
            all_files.extend(file_list)
        
        if not all_files:
            print("No parseable files found in data directory")
            return
        
        print(f"Found {len(all_files)} files to process in batch mode:")
        for f in all_files:
            print(f"  {f}")
        
        # Process each file
        for i, filepath in enumerate(all_files):
            process_single_file(filepath, cfg, all_grid_results, out_dir, is_first_file=(i==0))
        
        # Combine all grid results
        if all_grid_results:
            combined_results = pd.concat(all_grid_results, ignore_index=True)
            grid_file = out_dir / "grid_results.csv"
            combined_results.to_csv(grid_file, index=False)
            print(f"\nSaved combined grid results from {len(all_files)} files to {grid_file}")
        
    else:
        # Single file mode
        process_single_file(cfg.data_path, cfg, all_grid_results, out_dir, is_first_file=True)
        
        # Save/update grid results for single file
        if all_grid_results:
            grid_file = out_dir / "grid_results.csv"
            new_results = all_grid_results[0]
            filename = cfg.data_path.name
            
            # Load existing grid results or create new
            if grid_file.exists():
                try:
                    existing_results = pd.read_csv(grid_file)
                    # Remove existing entries for this file
                    existing_results = existing_results[existing_results['filename'] != filename]
                    # Combine with new results
                    combined_results = pd.concat([existing_results, new_results], ignore_index=True)
                except Exception as e:
                    print(f"Warning: Could not read existing grid results: {e}")
                    combined_results = new_results
            else:
                combined_results = new_results
            
            # Sort by filename for easy reading
            combined_results = combined_results.sort_values(['filename', 'kappa', 'c'])
            combined_results.to_csv(grid_file, index=False)
            print(f"Updated detailed grid results in {grid_file}")
    
    print(f"Updated summary results in {out_dir / 'summary_results.csv'}")


if __name__ == "__main__":
    from config import parse_args
    main(parse_args())
