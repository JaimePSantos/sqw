#!/usr/bin/env python3
"""
Debug script to investigate the mean files structure
"""

import os
import numpy as np
import pickle
from pathlib import Path

def debug_mean_files(base_dir="experiments_data_samples", N=2000):
    """Debug function to check what's in the mean files"""
    
    # Look for the no-noise case directory
    exp_dir = os.path.join(base_dir, "even_line_two_tesselation_angle_nonoise_0.0_0.0")
    
    if not os.path.exists(exp_dir):
        print(f"Directory not found: {exp_dir}")
        return
    
    print(f"Checking directory: {exp_dir}")
    
    # Check what files are available
    mean_files = []
    for filename in os.listdir(exp_dir):
        if filename.startswith("mean_step_") and filename.endswith(".pkl"):
            mean_files.append(filename)
    
    mean_files.sort()
    print(f"Found {len(mean_files)} mean files: {mean_files[:10]}...")  # Show first 10
    
    # Load first few mean files and analyze
    for i, filename in enumerate(mean_files[:5]):
        filepath = os.path.join(exp_dir, filename)
        with open(filepath, "rb") as f:
            mean_state = pickle.load(f)
        
        # Analyze the distribution
        mean_state_flat = mean_state.flatten()
        max_prob = np.max(mean_state_flat)
        max_pos = np.argmax(mean_state_flat)
        total_prob = np.sum(mean_state_flat)
        
        # Calculate center of mass
        domain = np.arange(len(mean_state_flat))
        center_of_mass = np.sum(domain * mean_state_flat) / total_prob if total_prob > 0 else 0
        
        # Calculate standard deviation
        variance = np.sum((domain - center_of_mass)**2 * mean_state_flat) / total_prob if total_prob > 0 else 0
        std = np.sqrt(variance)
        
        print(f"{filename}: shape={mean_state.shape}, max_prob={max_prob:.6f} at pos={max_pos}, "
              f"center_of_mass={center_of_mass:.2f}, std={std:.2f}, total_prob={total_prob:.6f}")
        
        # Check if this looks like an initial state (concentrated at center)
        center_pos = N // 2
        if center_pos < len(mean_state_flat):
            center_prob = mean_state_flat[center_pos]
            print(f"  Probability at center ({center_pos}): {center_prob:.6f}")

if __name__ == "__main__":
    debug_mean_files()
