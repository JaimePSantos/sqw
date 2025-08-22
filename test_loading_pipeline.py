#!/usr/bin/env python3
"""
Test the full smart loading pipeline to see where the issue occurs.
"""

import math
import os
import sys
import shutil

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_loading_static import (
    check_mean_probability_distributions_exist,
    load_mean_probability_distributions
)

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise"""
    return None

def test_smart_loading_pipeline():
    """Test the smart loading pipeline for different theta values"""
    
    # Test parameters - same as in main script
    N = 500
    steps = 500  
    samples = 10
    devs = [(0,0)]  # Only test the problematic (0,0) case
    
    # Test different theta values
    theta_values = [
        math.pi / 3,  # π/3 - where old data exists
        math.pi,      # π - current value
        math.pi / 4,  # π/4 - should not exist
    ]
    
    print("=== SMART LOADING PIPELINE TEST ===")
    print(f"N = {N}, steps = {steps}, samples = {samples}")
    print(f"Testing deviation: {devs[0]}")
    print()
    
    for theta in theta_values:
        print(f"Testing theta = {theta:.6f} ({theta/math.pi:.3f}π)")
        
        # Check if mean probability distributions exist
        exists = check_mean_probability_distributions_exist(
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            devs=devs,
            samples=samples,
            base_dir="experiments_data_samples_probDist",
            noise_type="static_noise",
            theta=theta
        )
        
        print(f"  Smart loading check result: {exists}")
        
        if exists:
            print("  Attempting to load existing data...")
            try:
                mean_results = load_mean_probability_distributions(
                    tesselation_func=dummy_tesselation_func,
                    N=N,
                    steps=steps,
                    devs=devs,
                    samples=samples,
                    base_dir="experiments_data_samples_probDist",
                    noise_type="static_noise",
                    theta=theta
                )
                
                if mean_results and len(mean_results) > 0 and mean_results[0] is not None:
                    # Get the first few time steps of the loaded data
                    dev_data = mean_results[0]  # Data for (0,0) deviation
                    if len(dev_data) >= 5:
                        print(f"  Loaded data sample (steps 0-4):")
                        for step in range(5):
                            if dev_data[step] is not None:
                                prob_dist = dev_data[step]
                                max_prob = max(prob_dist) if len(prob_dist) > 0 else 0
                                max_pos = list(prob_dist).index(max_prob) if len(prob_dist) > 0 else -1
                                print(f"    Step {step}: max_prob = {max_prob:.6f} at pos {max_pos}")
                            else:
                                print(f"    Step {step}: None (corrupted)")
                    else:
                        print(f"  ERROR: Insufficient data loaded (only {len(dev_data)} steps)")
                else:
                    print(f"  ERROR: Failed to load data or data is corrupted")
                    
            except Exception as e:
                print(f"  ERROR loading data: {e}")
        else:
            print("  No existing data found - would need to compute new data")
        
        print()

if __name__ == "__main__":
    test_smart_loading_pipeline()
