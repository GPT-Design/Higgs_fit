#!/usr/bin/env python3
"""
Utility script to list and explore available data files.
"""

import sys
from pathlib import Path
from file_parser import get_data_summary, FileParser


def main():
    """Main function to list data files."""
    parser = FileParser()
    
    if len(sys.argv) > 1:
        # Show detailed info for specific file
        filename = sys.argv[1]
        try:
            info = parser.get_file_summary(filename)
            print(f"File: {info['filename']}")
            print(f"Format: {info['format']}")
            print(f"Size: {info['size_bytes']} bytes")
            print(f"Parseable: {info['parseable']}")
            
            if info['parseable']:
                print(f"Data points: {info['data_points']}")
                print(f"Energy range: {info['energy_range'][0]:.2f} to {info['energy_range'][1]:.2f}")
                print(f"Observable range: {info['obs_range'][0]:.2e} to {info['obs_range'][1]:.2e}")
                
                # Show LIGO-specific metadata
                if 'detector' in info:
                    print(f"Detector: {info['detector']}")
                    print(f"Duration: {info['duration']} seconds")
                    print(f"GPS start: {info['gps_start']}")
                    print(f"Strain samples: {info['strain_samples']}")
            else:
                if 'parse_error' in info:
                    print(f"Parse error: {info['parse_error']}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Show summary of all files
        print("Available data files:")
        print("=" * 60)
        
        summary = get_data_summary()
        
        # Group by detector for LIGO files
        ligo_files = summary[summary['filename'].str.contains('-')]
        other_files = summary[~summary['filename'].str.contains('-')]
        
        if not other_files.empty:
            print("\nOther files:")
            print("-" * 40)
            for _, row in other_files.iterrows():
                status = "OK" if row['parseable'] else "ERR"
                print(f"{status} {row['filename']} ({row['format']})")
                if row['parseable']:
                    print(f"   {row['data_points']} points")
        
        if not ligo_files.empty:
            print("\nLIGO files:")
            print("-" * 40)
            
            # Group by detector
            detectors = {}
            for _, row in ligo_files.iterrows():
                detector = row['filename'].split('-')[0]
                if detector not in detectors:
                    detectors[detector] = []
                detectors[detector].append(row)
            
            for detector, files in detectors.items():
                detector_name = {'H': 'Hanford', 'L': 'Livingston', 'V': 'Virgo', 'G': 'GEO'}.get(detector, detector)
                print(f"\n{detector_name} ({detector}):")
                for file_info in files:
                    status = "OK" if file_info['parseable'] else "ERR"
                    print(f"  {status} {file_info['filename']}")
        
        print(f"\nTotal: {len(summary)} files")
        print(f"Parseable: {summary['parseable'].sum()}")
        print(f"Failed: {(~summary['parseable']).sum()}")
        
        print("\nUsage:")
        print("  python list_data.py                    # List all files")
        print("  python list_data.py <filename>         # Show file details")
        print("  python cli.py <filename>               # Run analysis")


if __name__ == "__main__":
    main()