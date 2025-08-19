#!/usr/bin/env python3

"""
Test script for the improved multiprocess archiving functionality.
This creates some dummy data and tests the archiving.
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add the current directory to the path so we can import from static_cluster_logged_mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_dummy_data(base_dir, N, num_folders=3):
    """Create dummy experiment data structure for testing"""
    print(f"Creating dummy data in {base_dir}...")
    
    # Create the experiments_data_samples structure
    data_dir = os.path.join(base_dir, "experiments_data_samples")
    os.makedirs(data_dir, exist_ok=True)
    
    # Create multiple experiment folders with N_xxx subfolders
    for i in range(num_folders):
        exp_dir = os.path.join(data_dir, f"dummy_tesselation_func_static_noise_{i}")
        theta_dir = os.path.join(exp_dir, "theta_1.047198")
        dev_dir = os.path.join(theta_dir, f"dev_max{i}.000_min0.500")
        n_dir = os.path.join(dev_dir, f"N_{N}")
        
        os.makedirs(n_dir, exist_ok=True)
        
        # Create some dummy files
        for j in range(3):
            step_dir = os.path.join(n_dir, f"step_{j}")
            os.makedirs(step_dir, exist_ok=True)
            
            dummy_file = os.path.join(step_dir, f"final_step_{j}_sample0.pkl")
            with open(dummy_file, 'w') as f:
                f.write(f"dummy data for step {j} in folder {i}")
    
    print(f"Created {num_folders} experiment folders with N_{N} data")
    return data_dir

def test_archiving():
    """Test the archiving functionality"""
    # Import the archiving functions from our script
    from static_cluster_logged_mp import create_experiment_archive, create_single_archive
    
    print("=== TESTING MULTIPROCESS ARCHIVING ===")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Save current directory and change to temp dir
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Test parameters
            N = 15000
            samples = 5
            
            # Create dummy data
            data_dir = create_dummy_data(temp_dir, N)
            
            print(f"\nTesting single-process archiving...")
            archive_name_single = create_experiment_archive(N, samples, use_multiprocess=False)
            if archive_name_single:
                print(f"✓ Single-process archive created: {archive_name_single}")
                size = os.path.getsize(archive_name_single) / 1024
                print(f"  Size: {size:.1f} KB")
            else:
                print("✗ Single-process archiving failed")
            
            print(f"\nTesting multiprocess archiving...")
            archive_name_multi = create_experiment_archive(N, samples, use_multiprocess=True, max_archive_processes=2)
            if archive_name_multi:
                print(f"✓ Multiprocess archive created: {archive_name_multi}")
                size = os.path.getsize(archive_name_multi) / 1024
                print(f"  Size: {size:.1f} KB")
            else:
                print("✗ Multiprocess archiving failed")
            
            print(f"\nTesting with no data (should handle gracefully)...")
            # Remove the data directory
            if os.path.exists("experiments_data_samples"):
                shutil.rmtree("experiments_data_samples")
            
            archive_name_nodata = create_experiment_archive(N, samples, use_multiprocess=True)
            if archive_name_nodata is None:
                print("✓ Correctly handled missing data directory")
            else:
                print("✗ Should have returned None for missing data")
                
        finally:
            # Restore original directory
            os.chdir(original_dir)
    
    print("\n=== ARCHIVING TESTS COMPLETED ===")

if __name__ == "__main__":
    test_archiving()
