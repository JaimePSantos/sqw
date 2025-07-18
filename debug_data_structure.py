#!/usr/bin/env python3
"""
Debug script to investigate the actual structure of the experimental data files.
"""

import os
import pickle
import numpy as np
from pathlib import Path

def inspect_data_files():
    """Inspect the actual data files to understand their structure."""
    
    # Check the no-noise case first
    base_dir = "experiments_data_samples"
    exp_dir = os.path.join(base_dir, "even_line_two_tesselation_angle_nonoise_0_0")
    
    print(f"Inspecting directory: {exp_dir}")
    
    if not os.path.exists(exp_dir):
        print(f"Directory does not exist: {exp_dir}")
        return
    
    # List all files in the directory
    print("\nFiles in directory:")
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, exp_dir)
            print(f"  {rel_path}")
    
    # Check a few mean files
    print("\nInspecting mean files:")
    for step in range(5):
        mean_file = os.path.join(exp_dir, f"mean_step_{step}.pkl")
        if os.path.exists(mean_file):
            with open(mean_file, "rb") as f:
                data = pickle.load(f)
            
            print(f"\nStep {step}:")
            print(f"  Type: {type(data)}")
            print(f"  Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            print(f"  Data type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
            print(f"  Min: {np.min(data):.6f}")
            print(f"  Max: {np.max(data):.6f}")
            print(f"  Sum: {np.sum(data):.6f}")
            print(f"  Mean: {np.mean(data):.6f}")
            
            # Check the distribution
            data_flat = data.flatten()
            non_zero_count = np.count_nonzero(data_flat)
            print(f"  Non-zero elements: {non_zero_count}/{len(data_flat)}")
            
            # Find peaks
            max_indices = np.where(data_flat == np.max(data_flat))[0]
            print(f"  Max value positions: {max_indices[:10]}")  # First 10 positions
            
            # Check if it's properly normalized
            if abs(np.sum(data) - 1.0) < 1e-10:
                print(f"  ✓ Properly normalized")
            else:
                print(f"  ✗ NOT properly normalized (sum={np.sum(data):.6f})")
    
    # Check if noise cases exist and are different
    print("\nChecking noise cases:")
    noise_dirs = [
        "even_line_two_tesselation_angle_noise_0.41887902047863906_0.41887902047863906",
        "even_line_two_tesselation_angle_noise_0.5235987755982988_0.5235987755982988"
    ]
    
    for noise_dir in noise_dirs:
        full_path = os.path.join(base_dir, noise_dir)
        if os.path.exists(full_path):
            print(f"\n{noise_dir}:")
            mean_file = os.path.join(full_path, "mean_step_0.pkl")
            if os.path.exists(mean_file):
                with open(mean_file, "rb") as f:
                    data = pickle.load(f)
                print(f"  Step 0 - Sum: {np.sum(data):.6f}, Max: {np.max(data):.6f}")
                
                # Compare with no-noise case
                no_noise_file = os.path.join(exp_dir, "mean_step_0.pkl")
                if os.path.exists(no_noise_file):
                    with open(no_noise_file, "rb") as f:
                        no_noise_data = pickle.load(f)
                    
                    if np.array_equal(data, no_noise_data):
                        print(f"  ✗ IDENTICAL to no-noise case!")
                    else:
                        print(f"  ✓ Different from no-noise case")
                        diff = np.abs(data - no_noise_data)
                        print(f"  Maximum difference: {np.max(diff):.6f}")
            else:
                print(f"  No mean_step_0.pkl file found")
        else:
            print(f"  Directory not found: {full_path}")

def check_sample_files():
    """Check individual sample files to understand the averaging process."""
    
    base_dir = "experiments_data_samples"
    exp_dir = os.path.join(base_dir, "even_line_two_tesselation_angle_nonoise_0_0")
    
    print(f"\nChecking sample files in: {exp_dir}")
    
    # Check step 0 samples
    step_dir = os.path.join(exp_dir, "step_0")
    if os.path.exists(step_dir):
        print(f"\nStep 0 samples:")
        sample_files = [f for f in os.listdir(step_dir) if f.startswith("final_step_0_sample")]
        print(f"Found {len(sample_files)} sample files")
        
        for i, sample_file in enumerate(sample_files[:3]):  # Check first 3 samples
            sample_path = os.path.join(step_dir, sample_file)
            with open(sample_path, "rb") as f:
                data = pickle.load(f)
            
            print(f"  Sample {i} ({sample_file}):")
            print(f"    Type: {type(data)}")
            print(f"    Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            # For quantum states, we need to convert to probabilities
            if hasattr(data, 'dtype') and np.iscomplexobj(data):
                prob_data = np.abs(data) ** 2
                print(f"    Complex quantum state, converted to probabilities")
                print(f"    Probability sum: {np.sum(prob_data):.6f}")
                print(f"    Max probability: {np.max(prob_data):.6f}")
            else:
                print(f"    Already probabilities")
                print(f"    Sum: {np.sum(data):.6f}")
                print(f"    Max: {np.max(data):.6f}")
    else:
        print(f"Step 0 directory not found: {step_dir}")

if __name__ == "__main__":
    inspect_data_files()
    check_sample_files()
