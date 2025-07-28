# version 0.2
"""
Batch fit driver: loop over every file and call fit.py programmatically.
Appends to a master fit_results_master.csv file.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from config import Config
from fit import fit_single_file
from file_parser import list_parseable_files


def batch_fit_all_files(cfg: Config, master_output: Path = None) -> None:
    """
    Run continuous fitting on all data files and save to master results.
    
    Args:
        cfg: Configuration object with fitting parameters
        master_output: Path for master results file (default: results/fit_results_master.csv)
    """
    if master_output is None:
        master_output = Path(cfg.output) / "fit_results_master.csv"
    
    # Get all parseable files
    files_by_type = list_parseable_files()
    all_files = []
    for file_list in files_by_type.values():
        all_files.extend(file_list)
    
    if not all_files:
        print("No parseable files found in data directory")
        return
    
    print(f"Starting batch fitting of {len(all_files)} files...")
    print(f"Master results will be saved to: {master_output}")
    
    # Load existing master results if available
    if master_output.exists():
        try:
            master_df = pd.read_csv(master_output)
            processed_files = set(master_df['filename'].tolist())
            print(f"Found existing master results with {len(processed_files)} files")
        except Exception as e:
            print(f"Warning: Could not read existing master results: {e}")
            master_df = pd.DataFrame()
            processed_files = set()
    else:
        master_df = pd.DataFrame()
        processed_files = set()
    
    # Process each file
    new_results = []
    failed_files = []
    skipped_files = []
    
    for i, filepath in enumerate(all_files, 1):
        filename = filepath.name
        
        # Skip if already processed
        if filename in processed_files:
            print(f"[{i:3d}/{len(all_files)}] Skipping {filename} (already processed)")
            skipped_files.append(filename)
            continue
        
        print(f"[{i:3d}/{len(all_files)}] Fitting {filename}...")
        
        try:
            # Run continuous fit
            result = fit_single_file(filepath, cfg, use_mcmc=False)
            new_results.append(result)
            
            print(f"  → kappa={result['kappa']:.4f}±{result.get('kappa_err', 0):.4f}, "
                  f"c={result['c']:.4f}±{result.get('c_err', 0):.4f}, "
                  f"logL={result['logL']:.3f}")
            
        except Exception as e:
            print(f"  → FAILED: {e}")
            failed_files.append((filename, str(e)))
            continue
    
    # Combine results
    if new_results:
        new_df = pd.DataFrame(new_results)
        if not master_df.empty:
            master_df = pd.concat([master_df, new_df], ignore_index=True)
        else:
            master_df = new_df
        
        # Sort by filename and save
        master_df = master_df.sort_values('filename')
        master_df.to_csv(master_output, index=False)
        print(f"\nMaster results updated: {len(new_results)} new fits added")
    
    # Summary
    print(f"\nBatch fitting summary:")
    print(f"  Total files: {len(all_files)}")
    print(f"  Successfully fit: {len(new_results)}")
    print(f"  Already processed: {len(skipped_files)}")
    print(f"  Failed: {len(failed_files)}")
    
    if failed_files:
        print(f"\nFailed files:")
        for filename, error in failed_files:
            print(f"  {filename}: {error}")
    
    if master_df.empty:
        print("No results to save.")
    else:
        total_results = len(master_df)
        print(f"\nMaster results file contains {total_results} total fits")
        print(f"Saved to: {master_output}")


def main():
    """CLI interface for batch fitting."""
    import argparse
    
    parser = argparse.ArgumentParser("Batch continuous fitting for all data files")
    parser.add_argument("--output", type=Path, default=Path("results"),
                       help="Output directory for results")
    parser.add_argument("--master-file", type=str, default="fit_results_master.csv",
                       help="Name of master results file")
    
    # Fitting parameters
    parser.add_argument("--alpha", type=float, default=0.04)
    parser.add_argument("--S", dest="S_scalar", type=float, default=0.10)
    parser.add_argument("--kappa-range", nargs=2, type=float, default=(0.0, 2.0))
    parser.add_argument("--c-range", nargs=2, type=float, default=(0.0, 1.5))
    parser.add_argument("--entropy-shape", default="log", 
                       choices=["constant", "powerlaw", "log"])
    
    args = parser.parse_args()
    
    # Create config
    cfg = Config(
        batch_mode=False,  # We handle batching ourselves
        output=args.output,
        alpha=args.alpha,
        S_scalar=args.S_scalar,
        kappa_min=args.kappa_range[0],
        kappa_max=args.kappa_range[1],
        c_min=args.c_range[0],
        c_max=args.c_range[1],
        entropy_shape=args.entropy_shape,
    )
    
    # Set up output directory
    cfg.output.mkdir(parents=True, exist_ok=True)
    master_output = cfg.output / args.master_file
    
    # Run batch fitting
    batch_fit_all_files(cfg, master_output)


if __name__ == "__main__":
    main()