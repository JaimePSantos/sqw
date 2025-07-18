#!/usr/bin/env python3
"""
Script to process individual sample files and create mean probability distributions.
This script goes through the sample files, converts quantum states to probabilities,
and calculates the mean probability distribution for each step and deviation.
"""

import os
import pickle
import numpy as np
from pathlib import Path

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="angle", base_dir="experiments_data"):
    """Get the experiment directory path."""
    func_name = tesselation_func.__name__ if hasattr(tesselation_func, '__name__') else 'unknown'
    noise_str = "noise" if has_noise else "nonoise" 
    param_str = "_".join(map(str, noise_params)) if noise_params else "0"
    return os.path.join(base_dir, f"{func_name}_{noise_type}_{noise_str}_{param_str}")

def amp2prob(state):
    """Convert quantum state amplitude to probability distribution."""
    return np.abs(state) ** 2

def process_experiments_to_prob_distributions(
    input_base_dir="experiments_data_samples",
    output_base_dir="experiments_data_samples_probDist",
    N=2000,
    steps=500,
    samples=10,
    tesselation_func_name="even_line_two_tesselation"
):
    """
    Process all experiment sample files and create mean probability distributions.
    
    Parameters
    ----------
    input_base_dir : str
        Base directory containing the original sample files
    output_base_dir : str
        Base directory to save the mean probability distributions
    N : int
        System size
    steps : int
        Number of time steps
    samples : int
        Number of samples per deviation
    tesselation_func_name : str
        Name of the tesselation function
    """
    
    # Create a mock tesselation function for directory naming
    class MockTesselationFunc:
        def __init__(self, name):
            self.__name__ = name
    
    tesselation_func = MockTesselationFunc(tesselation_func_name)
    
    # Define deviation values (same as in the original script)
    devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5, (np.pi/3) * 2]
    
    print(f"Processing experiments from {input_base_dir} to {output_base_dir}")
    print(f"Parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Deviations: {[f'{dev:.3f}' for dev in devs]}")
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each deviation
    for dev_idx, dev in enumerate(devs):
        has_noise = dev > 0
        
        # Get input and output directories
        input_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                         noise_params=[dev, dev], noise_type="angle", 
                                         base_dir=input_base_dir)
        output_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                          noise_params=[dev, dev], noise_type="angle", 
                                          base_dir=output_base_dir)
        
        print(f"\nProcessing dev {dev_idx} (angle_dev={dev:.3f})")
        print(f"  Input dir: {input_exp_dir}")
        print(f"  Output dir: {output_exp_dir}")
        
        if not os.path.exists(input_exp_dir):
            print(f"  Warning: Input directory not found, skipping")
            continue
        
        # Create output directory
        os.makedirs(output_exp_dir, exist_ok=True)
        
        # Process each step
        steps_processed = 0
        steps_failed = 0
        
        for step_idx in range(steps):
            step_dir = os.path.join(input_exp_dir, f"step_{step_idx}")
            
            if not os.path.exists(step_dir):
                if step_idx < 10:  # Only warn for first few steps
                    print(f"    Warning: Step directory not found: {step_dir}")
                steps_failed += 1
                continue
            
            # Load all samples for this step
            sample_prob_distributions = []
            samples_found = 0
            samples_missing = 0
            
            for sample_idx in range(samples):
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            quantum_state = pickle.load(f)
                        
                        # Convert to probability distribution
                        prob_dist = amp2prob(quantum_state)
                        sample_prob_distributions.append(prob_dist)
                        samples_found += 1
                        
                    except Exception as e:
                        print(f"    Error loading {filepath}: {e}")
                        samples_missing += 1
                else:
                    samples_missing += 1
            
            # Calculate mean probability distribution if we have samples
            if sample_prob_distributions:
                mean_prob_dist = np.mean(sample_prob_distributions, axis=0)
                
                # Save mean probability distribution
                mean_filename = f"mean_step_{step_idx}.pkl"
                mean_filepath = os.path.join(output_exp_dir, mean_filename)
                
                with open(mean_filepath, "wb") as f:
                    pickle.dump(mean_prob_dist, f)
                
                steps_processed += 1
                
                # Debug info for first few steps
                if step_idx < 5:
                    prob_sum = np.sum(mean_prob_dist)
                    max_prob = np.max(mean_prob_dist)
                    max_pos = np.argmax(mean_prob_dist.flatten())
                    print(f"    Step {step_idx}: {samples_found}/{samples} samples, "
                          f"prob_sum={prob_sum:.6f}, max_prob={max_prob:.6f} at pos={max_pos}")
            else:
                print(f"    Step {step_idx}: No valid samples found ({samples_missing} missing)")
                steps_failed += 1
        
        print(f"  Processed {steps_processed} steps successfully, {steps_failed} failed")
        
        # Create a summary file for this deviation
        summary = {
            'dev': dev,
            'steps_processed': steps_processed,
            'steps_failed': steps_failed,
            'samples_per_step': samples,
            'N': N
        }
        
        summary_filepath = os.path.join(output_exp_dir, "processing_summary.pkl")
        with open(summary_filepath, "wb") as f:
            pickle.dump(summary, f)
    
    print(f"\nProcessing complete! Mean probability distributions saved to {output_base_dir}")

def verify_mean_distributions(
    base_dir="experiments_data_samples_probDist",
    N=2000,
    steps=500,
    tesselation_func_name="even_line_two_tesselation",
    check_steps=10
):
    """
    Verify that the mean probability distributions were created correctly.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing the mean probability distributions
    N : int
        System size
    steps : int
        Number of time steps
    tesselation_func_name : str
        Name of the tesselation function
    check_steps : int
        Number of steps to verify in detail
    """
    
    # Create a mock tesselation function for directory naming
    class MockTesselationFunc:
        def __init__(self, name):
            self.__name__ = name
    
    tesselation_func = MockTesselationFunc(tesselation_func_name)
    
    # Define deviation values
    devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5, (np.pi/3) * 2]
    
    print(f"Verifying mean probability distributions in {base_dir}")
    
    for dev_idx, dev in enumerate(devs):
        has_noise = dev > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                   noise_params=[dev, dev], noise_type="angle", 
                                   base_dir=base_dir)
        
        print(f"\nDev {dev_idx} (angle_dev={dev:.3f}): {exp_dir}")
        
        if not os.path.exists(exp_dir):
            print(f"  ✗ Directory not found")
            continue
        
        # Check summary file
        summary_filepath = os.path.join(exp_dir, "processing_summary.pkl")
        if os.path.exists(summary_filepath):
            with open(summary_filepath, "rb") as f:
                summary = pickle.load(f)
            print(f"  Summary: {summary['steps_processed']} steps processed, {summary['steps_failed']} failed")
        
        # Check mean files
        mean_files_found = 0
        mean_files_missing = 0
        
        for step_idx in range(min(check_steps, steps)):
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                mean_files_found += 1
                
                # Load and check the file
                try:
                    with open(mean_filepath, "rb") as f:
                        mean_prob_dist = pickle.load(f)
                    
                    prob_sum = np.sum(mean_prob_dist)
                    max_prob = np.max(mean_prob_dist)
                    shape = mean_prob_dist.shape
                    
                    if step_idx < 5:  # Show details for first few steps
                        print(f"    Step {step_idx}: shape={shape}, sum={prob_sum:.6f}, max={max_prob:.6f}")
                    
                    # Check if properly normalized (should be close to 1.0 after normalization)
                    if prob_sum > 0:
                        normalized_sum = np.sum(mean_prob_dist / prob_sum)
                        if abs(normalized_sum - 1.0) > 1e-10:
                            print(f"    Warning: Step {step_idx} normalization issue")
                    
                except Exception as e:
                    print(f"    Error loading step {step_idx}: {e}")
                    mean_files_missing += 1
            else:
                mean_files_missing += 1
        
        print(f"  Mean files: {mean_files_found} found, {mean_files_missing} missing (checked first {check_steps} steps)")

def main():
    """Main function to process experiments and verify results."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sample files to create mean probability distributions')
    parser.add_argument('--input-dir', default='experiments_data_samples', 
                        help='Input directory containing sample files')
    parser.add_argument('--output-dir', default='experiments_data_samples_probDist',
                        help='Output directory for mean probability distributions')
    parser.add_argument('--N', type=int, default=2000, 
                        help='System size')
    parser.add_argument('--steps', type=int, default=500, 
                        help='Number of time steps')
    parser.add_argument('--samples', type=int, default=10, 
                        help='Number of samples per deviation')
    parser.add_argument('--tesselation-func', default='even_line_two_tesselation',
                        help='Name of tesselation function')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify existing mean distributions, do not process')
    
    args = parser.parse_args()
    
    if args.verify_only:
        print("Verification mode: checking existing mean distributions")
        verify_mean_distributions(
            base_dir=args.output_dir,
            N=args.N,
            steps=args.steps,
            tesselation_func_name=args.tesselation_func
        )
    else:
        print("Processing mode: creating mean probability distributions")
        process_experiments_to_prob_distributions(
            input_base_dir=args.input_dir,
            output_base_dir=args.output_dir,
            N=args.N,
            steps=args.steps,
            samples=args.samples,
            tesselation_func_name=args.tesselation_func
        )
        
        print("\nVerifying results...")
        verify_mean_distributions(
            base_dir=args.output_dir,
            N=args.N,
            steps=args.steps,
            tesselation_func_name=args.tesselation_func
        )

if __name__ == "__main__":
    main()
