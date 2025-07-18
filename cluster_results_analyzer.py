#!/usr/bin/env python3
"""
Cluster Results Analyzer - Load and analyze results from cluster quantum walk experiments.
This script handles loading experiment results from the cluster and performing statistical analysis.
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="angle", base_dir="experiments_data"):
    """Get the experiment directory path."""
    func_name = tesselation_func.__name__ if hasattr(tesselation_func, '__name__') else 'unknown'
    noise_str = "noise" if has_noise else "nonoise" 
    param_str = "_".join(map(str, noise_params)) if noise_params else "0"
    return os.path.join(base_dir, f"{func_name}_{noise_type}_{noise_str}_{param_str}")

def load_experiment_results(
    tesselation_func,
    N,
    steps,
    devs,
    samples,
    base_dir="experiments_data_samples"
):
    """
    Load all final states from disk for each dev with multiple samples.
    Returns: List[List[List]] - [dev][sample][step] -> state
    """
    results = []
    for dev in devs:
        has_noise = dev > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
        
        dev_results = []
        for sample_idx in range(samples):
            sample_states = []
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)
                    sample_states.append(state)
                else:
                    print(f"Warning: File not found: {filepath}")
                    sample_states.append(None)
            dev_results.append(sample_states)
        results.append(dev_results)
    return results

def load_mean_results(
    tesselation_func,
    N,
    steps,
    devs,
    base_dir="experiments_data_samples"
):
    """
    Load the mean probability distributions for each deviation and step.
    Returns: List[List] - [dev][step] -> mean_probability_distribution
    
    Note: These are mean probability distributions (|amplitude|²), not quantum states.
    """
    results = []
    for dev_idx, dev in enumerate(devs):
        has_noise = dev > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
        
        print(f"Loading mean results for dev {dev_idx} (dev={dev:.3f}) from: {exp_dir}")
        
        dev_results = []
        for step_idx in range(steps):
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                with open(mean_filepath, "rb") as f:
                    mean_state = pickle.load(f)
                dev_results.append(mean_state)
                
                # Debug: Check first few steps for Dev 0
                if dev_idx == 0 and step_idx < 3:
                    print(f"  Step {step_idx}: shape={mean_state.shape}, sum={np.sum(mean_state):.6f}, max={np.max(mean_state):.6f}")
            else:
                print(f"Warning: Mean file not found: {mean_filepath}")
                dev_results.append(None)
        results.append(dev_results)
    return results

def prob_distributions2std(prob_distributions, domain):
    """
    Calculate standard deviation from probability distributions.
    This should show spreading from an initial localized state.
    
    Parameters
    ----------
    prob_distributions : list
        List of probability distributions (already |amplitude|²)
    domain : array-like
        Position domain (e.g., np.arange(N))
        
    Returns
    -------
    list
        Standard deviations for each time step
    """
    std_values = []
    
    for step_idx, prob_dist in enumerate(prob_distributions):
        if prob_dist is None:
            std_values.append(0)
            continue
            
        # Ensure probability distribution is properly formatted
        prob_dist_flat = prob_dist.flatten()
        total_prob = np.sum(prob_dist_flat)
        
        if total_prob == 0:
            std_values.append(0)
            continue
            
        # Always normalize to ensure proper probability distribution
        # This handles the case where quantum states are not properly normalized
        prob_dist_flat = prob_dist_flat / total_prob
        
        # Calculate mean position (center of mass)
        mean_pos = np.sum(domain * prob_dist_flat)
        
        # Calculate variance (second moment about the mean)
        variance = np.sum((domain - mean_pos) ** 2 * prob_dist_flat)
        
        # Calculate standard deviation
        std = np.sqrt(variance) if variance > 0 else 0
        
        # Debug for first few steps
        if step_idx < 5:
            max_prob = np.max(prob_dist_flat)
            max_pos = np.argmax(prob_dist_flat)
            print(f"    Step {step_idx}: original_sum={total_prob:.6f}, normalized_sum={np.sum(prob_dist_flat):.6f}, mean_pos={mean_pos:.2f}, std={std:.2f}, max_prob={max_prob:.6f} at pos={max_pos}")
        
        std_values.append(std)
        
    return std_values

def plot_std_vs_time(stds, devs, N, steps, title_prefix="Angle noise", parameter_name="dev"):
    """
    Plot standard deviation vs time for different parameter values.
    Only plots the Dev 0 case (no noise).
    
    Parameters
    ----------
    stds : list
        List of standard deviation arrays for each parameter value
    devs : list
        List of parameter values
    N : int
        System size
    steps : int
        Number of time steps
    title_prefix : str
        Prefix for plot title
    parameter_name : str
        Name of the parameter for legend
    """
    plt.figure(figsize=(12, 8))
    
    time_steps = np.arange(steps)
    
    # Only plot Dev 0 (first element, no noise case)
    if len(stds) > 0 and stds[0]:
        dev_0_std = stds[0]
        dev_0_value = devs[0]
        plt.plot(time_steps, dev_0_std, 'o-', label=f'{parameter_name}={dev_0_value:.3f} (No Noise)', markersize=4, color='blue')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Standard Deviation')
    plt.title(f'{title_prefix} - Standard Deviation vs Time (N={N}) - No Noise Case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'std_vs_time_{title_prefix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_final_distributions(mean_results, devs, N, final_step_idx=-1, title_prefix="Angle noise", parameter_name="dev"):
    """
    Plot final probability distributions for different parameter values.
    Only plots the Dev 0 case (no noise).
    
    Parameters
    ----------
    mean_results : list
        List of mean probability distributions for each parameter value
    devs : list
        List of parameter values
    N : int
        System size
    final_step_idx : int
        Index of the final step to plot (default: -1 for last step)
    title_prefix : str
        Prefix for plot title
    parameter_name : str
        Name of the parameter for legend
    """
    plt.figure(figsize=(12, 8))
    
    positions = np.arange(N)
    
    # Only plot Dev 0 (first element, no noise case)
    if len(mean_results) > 0 and mean_results[0] and len(mean_results[0]) > 0:
        dev_0_results = mean_results[0]
        dev_0_value = devs[0]
        final_dist = dev_0_results[final_step_idx]
        if final_dist is not None:
            plt.plot(positions, final_dist.flatten(), 'o-', label=f'{parameter_name}={dev_0_value:.3f} (No Noise)', markersize=2, color='blue')
    
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title(f'{title_prefix} - Final Probability Distribution (N={N}) - No Noise Case')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'final_distribution_{title_prefix.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_cluster_results(
    base_dir="experiments_data_samples_probDist",
    N=2000,
    steps=None,
    samples=10,
    devs=None,
    tesselation_func_name="even_line_two_tesselation"
):
    """
    Main analysis function for cluster results.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment results
    N : int
        System size
    steps : int
        Number of time steps (if None, will be calculated as N//4)
    samples : int
        Number of samples per deviation
    devs : list
        List of deviation values (if None, will use default)
    tesselation_func_name : str
        Name of the tesselation function used
    """
    if steps is None:
        steps = N // 4
    
    if devs is None:
        devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5, (np.pi/3) * 2]
    
    # Create a mock tesselation function for directory naming
    class MockTesselationFunc:
        def __init__(self, name):
            self.__name__ = name
    
    tesselation_func = MockTesselationFunc(tesselation_func_name)
    
    print(f"Analyzing results from {base_dir}")
    print(f"Parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Deviations: {devs}")
    
    # Load mean results
    print("Loading mean results...")
    mean_results = load_mean_results(
        tesselation_func=tesselation_func,
        N=N,
        steps=steps,
        devs=devs,
        base_dir=base_dir
    )
    
    # Calculate statistics
    print("Calculating statistics...")
    domain = np.arange(N)
    stds = []
    
    for i, dev_mean_prob_dists in enumerate(mean_results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Dev {i} (angle_dev={devs[i]:.3f}): {len(std_values)} std values")
            
            # Debug: Check the first few time steps for Dev 0
            if i == 0:
                print(f"Debug Dev 0 - First 5 time steps:")
                for step_idx in range(min(5, len(dev_mean_prob_dists))):
                    prob_dist = dev_mean_prob_dists[step_idx]
                    if prob_dist is not None:
                        # Find where probability is concentrated
                        max_prob_idx = np.argmax(prob_dist)
                        max_prob = np.max(prob_dist)
                        mean_pos = np.sum(domain * prob_dist.flatten())
                        std_val = std_values[step_idx]
                        print(f"  Step {step_idx}: max_prob={max_prob:.6f} at pos={max_prob_idx}, mean_pos={mean_pos:.2f}, std={std_val:.2f}")
                        
                        # Check if this looks like initial state
                        if step_idx == 0:
                            center_pos = N // 2
                            center_prob = prob_dist[center_pos] if center_pos < len(prob_dist) else 0
                            center_prob_val = float(center_prob.flatten()[0]) if hasattr(center_prob, 'flatten') else float(center_prob)
                            print(f"    Initial state check: center_pos={center_pos}, center_prob={center_prob_val:.6f}")
        else:
            print(f"Dev {i} (angle_dev={devs[i]:.3f}): No valid mean probability distributions")
            stds.append([])
    
    # Create plots
    print("Creating plots...")
    plot_std_vs_time(stds, devs, N, steps, title_prefix="Angle noise", parameter_name="angle_dev")
    plot_final_distributions(mean_results, devs, N, final_step_idx=-1, title_prefix="Angle noise", parameter_name="angle_dev")
    
    print("Analysis complete!")
    
    return {
        'mean_results': mean_results,
        'stds': stds,
        'devs': devs,
        'N': N,
        'steps': steps,
        'samples': samples
    }

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze cluster quantum walk experiment results')
    parser.add_argument('--base-dir', default='experiments_data_samples_probDist', 
                        help='Base directory containing experiment results')
    parser.add_argument('--N', type=int, default=2000, 
                        help='System size')
    parser.add_argument('--steps', type=int, 
                        help='Number of time steps (default: N//4)')
    parser.add_argument('--samples', type=int, default=10, 
                        help='Number of samples per deviation')
    parser.add_argument('--devs', nargs='+', type=float,
                        help='List of deviation values')
    parser.add_argument('--tesselation-func', default='even_line_two_tesselation',
                        help='Name of tesselation function used')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_cluster_results(
        base_dir=args.base_dir,
        N=args.N,
        steps=args.steps,
        samples=args.samples,
        devs=args.devs,
        tesselation_func_name=args.tesselation_func
    )
    
    return results

if __name__ == "__main__":
    main()
