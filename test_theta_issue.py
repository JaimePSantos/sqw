#!/usr/bin/env python3
"""
Test script to demonstrate and fix the theta value issue.
This script will test directory resolution for different theta values.
"""

import math
import os
import sys

# Add the current directory to path to import smart_loading_static
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_loading_static import get_experiment_dir, check_mean_probability_distributions_exist

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise"""
    return None

def test_theta_directory_resolution():
    """Test how different theta values resolve to directories"""
    
    # Test parameters
    N = 500
    steps = 500
    samples = 10
    dev = (0, 0)  # (0,0) deviation that's causing issues
    
    # Test different theta values
    theta_values = [
        math.pi / 3,  # π/3 ≈ 1.047198
        math.pi,      # π ≈ 3.141593
        math.pi / 4,  # π/4 ≈ 0.785398
        2 * math.pi   # 2π ≈ 6.283185
    ]
    
    print("=== THETA DIRECTORY RESOLUTION TEST ===")
    
    for theta in theta_values:
        print(f"\nTesting theta = {theta:.6f} ({theta/math.pi:.3f}π)")
        
        # Test samples directory resolution
        samples_dir = get_experiment_dir(
            tesselation_func=dummy_tesselation_func,
            has_noise=False,  # (0,0) means no noise
            N=N,
            noise_params=[dev],
            noise_type="static_noise",
            base_dir="experiments_data_samples",
            theta=theta,
            samples=None
        )
        
        # Test probDist directory resolution  
        probdist_dir = get_experiment_dir(
            tesselation_func=dummy_tesselation_func,
            has_noise=False,  # (0,0) means no noise
            N=N,
            noise_params=[dev],
            noise_type="static_noise",
            base_dir="experiments_data_samples_probDist",
            theta=theta,
            samples=samples
        )
        
        print(f"  Samples dir:  {samples_dir}")
        print(f"  ProbDist dir: {probdist_dir}")
        
        # Check if directories exist
        samples_exists = os.path.exists(samples_dir)
        probdist_exists = os.path.exists(probdist_dir)
        
        print(f"  Samples exist:  {samples_exists}")
        print(f"  ProbDist exist: {probdist_exists}")
        
        # Test smart loading check
        if probdist_exists:
            exists_check = check_mean_probability_distributions_exist(
                tesselation_func=dummy_tesselation_func,
                N=N,
                steps=steps,
                devs=[dev],
                samples=samples,
                base_dir="experiments_data_samples_probDist",
                noise_type="static_noise",
                theta=theta
            )
            print(f"  Smart loading check: {exists_check}")

if __name__ == "__main__":
    test_theta_directory_resolution()
