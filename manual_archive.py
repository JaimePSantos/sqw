#!/usr/bin/env python3

"""
Manual archiving utility for experiment data.

This script allows you to manually create archives of experiment data,
especially useful when you want to archive data that was computed on a cluster
but didn't get archived during the original run.
"""

import os
import sys
import argparse
from datetime import datetime

# Add the current directory to the path so we can import from static_cluster_logged_mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Create archive of experiment data")
    parser.add_argument("N", type=int, help="System size (N value) to archive")
    parser.add_argument("samples", type=int, help="Number of samples per deviation")
    parser.add_argument("--multiprocess", "-m", action="store_true", 
                       help="Use multiprocess archiving (default: auto-detect based on folder count)")
    parser.add_argument("--processes", "-p", type=int, default=None,
                       help="Number of processes for multiprocess archiving (default: auto-detect)")
    parser.add_argument("--data-dir", "-d", default="experiments_data_samples",
                       help="Data directory to archive (default: experiments_data_samples)")
    
    args = parser.parse_args()
    
    print("=== MANUAL ARCHIVING UTILITY ===")
    print(f"Target N: {args.N}")
    print(f"Samples: {args.samples}")
    print(f"Data directory: {args.data_dir}")
    
    # Change to data directory if it exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' not found")
        print("Make sure you're running this script from the correct location")
        return 1
    
    # Import the archiving function
    try:
        from static_cluster_logged_mp import create_experiment_archive
    except ImportError as e:
        print(f"ERROR: Could not import archiving function: {e}")
        print("Make sure static_cluster_logged_mp.py is in the same directory")
        return 1
    
    # Determine multiprocess settings
    use_multiprocess = args.multiprocess
    max_processes = args.processes
    
    if not use_multiprocess and max_processes is None:
        # Auto-detect: use multiprocess if there are many folders
        n_folder_name = f"N_{args.N}"
        folder_count = 0
        for root, dirs, files in os.walk(args.data_dir):
            if n_folder_name in dirs:
                folder_count += 1
        
        if folder_count > 3:
            use_multiprocess = True
            print(f"Auto-detected {folder_count} folders - enabling multiprocess archiving")
        else:
            use_multiprocess = False
            print(f"Auto-detected {folder_count} folders - using single-process archiving")
    
    print(f"Multiprocess archiving: {use_multiprocess}")
    if use_multiprocess and max_processes:
        print(f"Max processes: {max_processes}")
    
    # Create the archive
    print("\nStarting archiving...")
    archive_name = create_experiment_archive(args.N, args.samples, use_multiprocess, max_processes)
    
    if archive_name:
        print(f"\n✓ SUCCESS: Archive created: {archive_name}")
        
        # Show archive info
        if os.path.exists(archive_name):
            size_mb = os.path.getsize(archive_name) / (1024 * 1024)
            print(f"Archive size: {size_mb:.1f} MB")
            print(f"Archive location: {os.path.abspath(archive_name)}")
        
        return 0
    else:
        print("\n✗ FAILED: Archive creation failed or no data found")
        return 1

if __name__ == "__main__":
    sys.exit(main())
