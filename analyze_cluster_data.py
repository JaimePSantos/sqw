#!/usr/bin/env python3
"""
Analysis script for existing cluster data.

This script runs analysis on the data that was already computed on the cluster,
skipping the sample computation phase that timed out.
"""

import os
import sys

# Set configuration to skip sample computation and just do analysis
os.environ['SKIP_SAMPLE_COMPUTATION'] = 'True'  # Skip samples, just analyze
os.environ['CALCULATE_SAMPLES_ONLY'] = 'False'  # We want full analysis
os.environ['ENABLE_PLOTTING'] = 'True'  # Enable plotting for analysis
os.environ['CREATE_TAR_ARCHIVE'] = 'True'  # Create archive of results
os.environ['USE_MULTIPROCESS_ARCHIVING'] = 'True'  # Use multiprocess archiving
os.environ['EXCLUDE_SAMPLES_FROM_ARCHIVE'] = 'True'  # Only archive processed results

# Set the same parameters as your cluster run
os.environ['FORCE_N_VALUE'] = '20000'
os.environ['FORCE_SAMPLES_COUNT'] = '5'

# Import and run the main script
if __name__ == "__main__":
    print("=== ANALYZING EXISTING CLUSTER DATA ===")
    print("Configuration:")
    print("  Sample computation: SKIPPED")
    print("  Analysis phase: ENABLED")
    print("  N = 20000 (from cluster)")
    print("  Samples = 5 (from cluster)")
    print("  Plotting: ENABLED")
    print("  Archiving: ENABLED")
    print("=" * 50)
    
    # Import and run the updated script
    from static_cluster_logged_mp import run_static_experiment
    
    try:
        result = run_static_experiment()
        print("\n=== ANALYSIS COMPLETED SUCCESSFULLY ===")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Total time: {result.get('total_time', 0):.1f} seconds")
        if 'archiving_enabled' in result:
            print(f"Archiving: {'ENABLED' if result['archiving_enabled'] else 'DISABLED'}")
    except Exception as e:
        print(f"\n=== ANALYSIS FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Print helpful debugging info
        print("\n=== DEBUGGING INFO ===")
        print("Checking for existing data directories...")
        
        data_dirs = [
            "experiments_data_samples",
            "experiments_data_samples_probDist", 
            "experiments_data_samples_std"
        ]
        
        for dir_name in data_dirs:
            if os.path.exists(dir_name):
                print(f"  ✓ {dir_name} exists")
                try:
                    subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d))]
                    print(f"    Contains {len(subdirs)} subdirectories")
                except:
                    print(f"    (Cannot list contents)")
            else:
                print(f"  ✗ {dir_name} missing")
