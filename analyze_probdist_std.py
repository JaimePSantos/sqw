#!/usr/bin/env python3
"""
Analyze probability distributions and plot standard deviation vs time.
This script loads mean probability distributions from experiments_data_samples_probDist,
calculates standard deviation for each time step, and plots the results.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from jaime_scripts import prob_distributions2std, get_experiment_dir

def load_mean_probability_distributions(exp_dir, steps):
    """
    Load mean probability distributions from a specific experiment directory.
    
    Parameters
    ----------
    exp_dir : str
        Path to experiment directory containing mean_step_X.pkl files
    steps : int
        Number of time steps to load
        
    Returns
    -------
    list
        List of mean probability distributions for each step
    """
    prob_distributions = []
    failed_loads = 0
    
    for step_idx in range(steps):
        mean_filename = f"mean_step_{step_idx}.pkl"
        mean_filepath = os.path.join(exp_dir, mean_filename)
        
        if os.path.exists(mean_filepath):
            try:
                with open(mean_filepath, "rb") as f:
                    mean_prob_dist = pickle.load(f)
                prob_distributions.append(mean_prob_dist)
            except Exception as e:
                failed_loads += 1
                if failed_loads <= 5:  # Only print first few failures
                    print(f"Warning: Could not load {mean_filename}: {e}")
                prob_distributions.append(None)
        else:
            prob_distributions.append(None)
    
    if failed_loads > 0:
        print(f"Total failed loads: {failed_loads}/{steps}")
    
    return prob_distributions

def find_experiment_dirs(base_dir):
    """
    Find all experiment directories in the probDist folder.
    
    Parameters
    ----------
    base_dir : str
        Base directory to search (experiments_data_samples_probDist)
        
    Returns
    -------
    list
        List of tuples (label, exp_dir) for each experiment
    """
    exp_dirs = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return exp_dirs
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(base_path):
        # Check if this directory contains mean_step_X.pkl files
        if any(f.startswith("mean_step_") and f.endswith(".pkl") for f in files):
            # Extract a meaningful label from the path
            rel_path = Path(root).relative_to(base_path)
            label = str(rel_path).replace(os.sep, "_")
            exp_dirs.append((label, root))
    
    return exp_dirs

def plot_std_vs_time(std_data, N, title="Standard Deviation vs Time"):
    """
    Plot standard deviation vs time for multiple experiments.
    
    Parameters
    ----------
    std_data : list
        List of tuples (label, std_values) for each experiment
    N : int
        System size (for plot title)
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for label, std_values in std_data:
        time_steps = np.arange(len(std_values))
        plt.plot(time_steps, std_values, 'o-', label=label, markersize=2, alpha=0.8)
    
    plt.xlabel('Time Steps')
    plt.ylabel('Standard Deviation')
    plt.title(f'{title} (N={N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'std_vs_time_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_final_distributions(prob_data, N, final_step_idx=-1, title="Final Probability Distributions"):
    """
    Plot final probability distributions for multiple experiments.
    
    Parameters
    ----------
    prob_data : list
        List of tuples (label, prob_distributions) for each experiment
    N : int
        System size
    final_step_idx : int
        Index of the final step to plot (default: -1 for last step)
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    positions = np.arange(N)
    
    for label, prob_distributions in prob_data:
        if prob_distributions and len(prob_distributions) > 0:
            final_dist = prob_distributions[final_step_idx]
            if final_dist is not None:
                plt.plot(positions, final_dist.flatten(), 'o-', label=label, markersize=1, alpha=0.8)
    
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title(f'{title} (N={N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'final_distributions_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main(base_dir="experiments_data_samples_probDist", N=2000, steps=500):
    """
    Main analysis function.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment results
    N : int
        System size
    steps : int
        Number of time steps
    """
    print(f"Analyzing probability distributions from {base_dir}")
    print(f"Parameters: N={N}, steps={steps}")
    
    # Find all experiment directories
    exp_dirs = find_experiment_dirs(base_dir)
    
    if not exp_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(exp_dirs)} experiment directories:")
    for label, exp_dir in exp_dirs:
        print(f"  - {label}: {exp_dir}")
    
    # Create domain for standard deviation calculation (centered around 0)
    domain = np.arange(N) - N//2
    
    # Load data and calculate statistics for each experiment
    std_data = []
    prob_data = []
    
    for label, exp_dir in exp_dirs:
        print(f"\nProcessing {label}...")
        
        # Load mean probability distributions
        prob_distributions = load_mean_probability_distributions(exp_dir, steps)
        
        # Calculate standard deviations
        std_values = prob_distributions2std(prob_distributions, domain)
        
        print(f"  Loaded {len([p for p in prob_distributions if p is not None])} valid distributions")
        print(f"  Calculated {len(std_values)} standard deviation values")
        
        # Store results
        std_data.append((label, std_values))
        prob_data.append((label, prob_distributions))
        
        # Print some statistics
        if std_values:
            print(f"  Initial std: {std_values[0]:.3f}")
            print(f"  Final std: {std_values[-1]:.3f}")
            print(f"  Max std: {max(std_values):.3f}")
    
    # Create plots
    print("\nCreating plots...")
    plot_std_vs_time(std_data, N, title="Standard Deviation vs Time - All Experiments")
    plot_final_distributions(prob_data, N, final_step_idx=-1, title="Final Probability Distributions - All Experiments")
    
    print("\nAnalysis complete!")
    print("Generated plots:")
    print("  - std_vs_time_analysis.png")
    print("  - final_distributions_analysis.png")
    
    return {
        'std_data': std_data,
        'prob_data': prob_data,
        'N': N,
        'steps': steps
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze probability distributions and plot standard deviation')
    parser.add_argument('--base-dir', default='experiments_data_samples_probDist', 
                        help='Base directory containing experiment results')
    parser.add_argument('--N', type=int, default=2000, 
                        help='System size')
    parser.add_argument('--steps', type=int, default=500, 
                        help='Number of time steps')
    
    args = parser.parse_args()
    
    # Run analysis
    results = main(
        base_dir=args.base_dir,
        N=args.N,
        steps=args.steps
    )
