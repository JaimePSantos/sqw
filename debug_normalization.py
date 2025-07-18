#!/usr/bin/env python3
"""
Debug script to specifically investigate the quantum state normalization issue.
"""

import os
import pickle
import numpy as np

def check_quantum_state_normalization():
    """Check if quantum states are properly normalized and what's happening during averaging."""
    
    # Check the no-noise case
    base_dir = "experiments_data_samples"
    exp_dir = os.path.join(base_dir, "even_line_two_tesselation_angle_nonoise_0_0")
    
    print("Checking quantum state normalization:")
    
    # Check step 0 samples
    step_dir = os.path.join(exp_dir, "step_0")
    if os.path.exists(step_dir):
        print(f"\nStep 0 samples:")
        
        sample_files = [f for f in os.listdir(step_dir) if f.startswith("final_step_0_sample")]
        sample_files.sort()
        
        all_prob_states = []
        
        for i, sample_file in enumerate(sample_files):
            sample_path = os.path.join(step_dir, sample_file)
            with open(sample_path, "rb") as f:
                quantum_state = pickle.load(f)
            
            # Convert to probabilities
            prob_state = np.abs(quantum_state) ** 2
            all_prob_states.append(prob_state)
            
            print(f"  Sample {i}:")
            print(f"    Quantum state sum: {np.sum(np.abs(quantum_state) ** 2):.6f}")
            print(f"    Max probability: {np.max(prob_state):.6f}")
            print(f"    Prob positions: {np.where(prob_state > 0.01)[0]}")
            
            # Check if quantum state is properly normalized
            if abs(np.sum(np.abs(quantum_state) ** 2) - 1.0) < 1e-10:
                print(f"    ✓ Quantum state properly normalized")
            else:
                print(f"    ✗ Quantum state NOT properly normalized!")
        
        # Calculate mean manually
        print(f"\nManual averaging:")
        mean_prob_state = np.mean(all_prob_states, axis=0)
        print(f"  Mean probability sum: {np.sum(mean_prob_state):.6f}")
        print(f"  Mean max probability: {np.max(mean_prob_state):.6f}")
        
        # Load the stored mean file
        mean_file = os.path.join(exp_dir, "mean_step_0.pkl")
        if os.path.exists(mean_file):
            with open(mean_file, "rb") as f:
                stored_mean = pickle.load(f)
            
            print(f"\nStored mean file:")
            print(f"  Stored mean sum: {np.sum(stored_mean):.6f}")
            print(f"  Stored mean max: {np.max(stored_mean):.6f}")
            
            # Check if they match
            if np.allclose(mean_prob_state, stored_mean):
                print(f"  ✓ Manual mean matches stored mean")
            else:
                print(f"  ✗ Manual mean does NOT match stored mean!")
                diff = np.abs(mean_prob_state - stored_mean)
                print(f"  Max difference: {np.max(diff):.6f}")
    
    # Check step 1 to see what's happening
    step_dir = os.path.join(exp_dir, "step_1")
    if os.path.exists(step_dir):
        print(f"\nStep 1 samples (first 3 only):")
        
        sample_files = [f for f in os.listdir(step_dir) if f.startswith("final_step_1_sample")]
        sample_files.sort()
        
        for i, sample_file in enumerate(sample_files[:3]):
            sample_path = os.path.join(step_dir, sample_file)
            with open(sample_path, "rb") as f:
                quantum_state = pickle.load(f)
            
            # Convert to probabilities
            prob_state = np.abs(quantum_state) ** 2
            
            print(f"  Sample {i}:")
            print(f"    Quantum state sum: {np.sum(np.abs(quantum_state) ** 2):.6f}")
            print(f"    Max probability: {np.max(prob_state):.6f}")
            
            # Check if quantum state is properly normalized
            if abs(np.sum(np.abs(quantum_state) ** 2) - 1.0) < 1e-10:
                print(f"    ✓ Quantum state properly normalized")
            else:
                print(f"    ✗ Quantum state NOT properly normalized!")

if __name__ == "__main__":
    check_quantum_state_normalization()
