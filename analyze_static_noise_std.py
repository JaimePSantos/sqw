#!/usr/bin/env python3
"""
Simple static noise standard deviation analysis.
Plots standard deviation vs time for static noise experiments given the parameters.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from jaime_scripts import prob_distributions2std

def load_static_noise_std_data(N, theta, deviation_ranges, steps, base_dir="experiments_data_samples_probDist"):
    """
    Load static noise probability distributions and calculate standard deviation vs time.
    
    Parameters
    ----------
    N : int
        System size
    theta : float
        Theta parameter
    deviation_ranges : list
        List of noise deviations to analyze
    steps : int
        Number of time steps
    base_dir : str
        Base directory containing experiment results
        
    Returns
    -------
    dict
        Dictionary with std_data for each deviation
    """
    std_data = {}
    domain = np.arange(N) - N//2  # Centered domain for std calculation
    
    base_path = Path(base_dir)
    
    print(f"Loading static noise data for N={N}, theta={theta:.4f}, steps={steps}")
    print(f"Looking in: {base_dir}")
    
    for deviation in deviation_ranges:
        print(f"\nProcessing deviation {deviation:.3f}...")
        
        # Find the correct experiment directory
        found_dir = None
        
        # Look for static noise experiment directories
        for exp_dir in base_path.iterdir():
            if exp_dir.is_dir() and "static_noise_tesselation" in exp_dir.name:
                
                # Handle zero deviation (nonoise)
                if deviation == 0.0 and "nonoise" in exp_dir.name:
                    # Look for N_X subdirectory
                    n_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name == f"N_{N}"]
                    if n_dirs:
                        found_dir = n_dirs[0]
                        break
                
                # Handle non-zero deviations
                elif deviation > 0.0 and "nonoise" not in exp_dir.name:
                    # Look for dev_X.XXX subdirectory
                    dev_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name == f"dev_{deviation:.3f}"]
                    for dev_dir in dev_dirs:
                        # Look for N_X subdirectory
                        n_dirs = [d for d in dev_dir.iterdir() if d.is_dir() and d.name == f"N_{N}"]
                        if n_dirs:
                            found_dir = n_dirs[0]
                            break
                    if found_dir:
                        break
        
        if not found_dir:
            print(f"  ⚠️  No data found for deviation {deviation:.3f}")
            continue
        
        print(f"  📂 Found directory: {found_dir}")
        
        # Load mean probability distributions for all time steps
        prob_distributions = []
        loaded_steps = 0
        
        for step_idx in range(steps):
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = found_dir / mean_filename
            
            if mean_filepath.exists():
                try:
                    with open(mean_filepath, "rb") as f:
                        mean_prob_dist = pickle.load(f)
                    prob_distributions.append(mean_prob_dist)
                    loaded_steps += 1
                except Exception as e:
                    print(f"    Warning: Could not load {mean_filename}: {e}")
                    prob_distributions.append(None)
            else:
                prob_distributions.append(None)
        
        print(f"  📊 Loaded {loaded_steps}/{steps} time steps")
        
        if loaded_steps > 0:
            # Calculate standard deviation evolution
            std_values = prob_distributions2std(prob_distributions, domain)
            std_data[f"dev_{deviation:.3f}"] = {
                'deviation': deviation,
                'std_values': std_values,
                'steps_loaded': loaded_steps,
                'prob_distributions': prob_distributions
            }
            print(f"  ✅ Calculated {len(std_values)} std values")
        else:
            print(f"  ❌ No valid data for deviation {deviation:.3f}")
    
    return std_data

def plot_static_noise_std_vs_time(std_data, N, theta, title="Static Noise: Standard Deviation vs Time"):
    """
    Plot standard deviation vs time for static noise experiments.
    
    Parameters
    ----------
    std_data : dict
        Dictionary with std data for each deviation
    N : int
        System size
    theta : float
        Theta parameter
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Sort by deviation for consistent plotting
    sorted_items = sorted(std_data.items(), key=lambda x: x[1]['deviation'])
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_items)))
    
    for i, (label, data) in enumerate(sorted_items):
        deviation = data['deviation']
        std_values = data['std_values']
        
        if std_values:
            time_steps = np.arange(len(std_values))
            plt.plot(time_steps, std_values, 'o-', 
                    color=colors[i], 
                    label=f'σ={deviation:.3f}', 
                    markersize=4, 
                    alpha=0.8,
                    linewidth=2)
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.title(f'{title}\n(N={N}, θ={theta:.4f})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filename = f'static_noise_std_vs_time_N{N}_theta{theta:.4f}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    
    plt.show()

def analyze_static_noise_std(N, theta, deviation_ranges, steps, base_dir="experiments_data_samples_probDist"):
    """
    Main function to analyze static noise standard deviation vs time.
    
    Parameters
    ----------
    N : int
        System size
    theta : float
        Theta parameter (e.g., np.pi/4)
    deviation_ranges : list
        List of noise deviations to analyze (e.g., [0.0, 0.1, 0.2, 0.3])
    steps : int
        Number of time steps
    base_dir : str
        Base directory containing experiment results
    """
    print("=" * 60)
    print("STATIC NOISE STANDARD DEVIATION ANALYSIS")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  N = {N}")
    print(f"  theta = {theta:.4f}")
    print(f"  deviation_ranges = {deviation_ranges}")
    print(f"  steps = {steps}")
    print(f"  base_dir = {base_dir}")
    print()
    
    # Load data and calculate std
    std_data = load_static_noise_std_data(N, theta, deviation_ranges, steps, base_dir)
    
    if not std_data:
        print("❌ No data found! Check your parameters and data directory.")
        return None
    
    print(f"\n✅ Successfully loaded data for {len(std_data)} deviations")
    
    # Create plot
    print("\n📈 Creating standard deviation vs time plot...")
    plot_static_noise_std_vs_time(std_data, N, theta)
    
    # Print summary
    print("\n📊 Summary:")
    for label, data in sorted(std_data.items(), key=lambda x: x[1]['deviation']):
        dev = data['deviation']
        std_vals = data['std_values']
        if std_vals:
            print(f"  σ={dev:.3f}: {len(std_vals)} time steps, "
                  f"initial_std={std_vals[0]:.3f}, final_std={std_vals[-1]:.3f}")
    
    print("\n✅ Analysis complete!")
    return std_data

def quick_static_noise_std_plot(N, theta=np.pi/4, deviation_ranges=None, steps=20):
    """
    Quick function to plot static noise std vs time with minimal parameters.
    
    Parameters
    ----------
    N : int
        System size
    theta : float, optional
        Theta parameter (default: π/4)
    deviation_ranges : list, optional
        List of deviations (default: [0.0, 0.1, 0.2, 0.3])
    steps : int, optional
        Number of time steps (default: 20)
    
    Example
    -------
    >>> quick_static_noise_std_plot(200, deviation_ranges=[0.0, 0.1, 0.3, 0.5])
    """
    if deviation_ranges is None:
        deviation_ranges = [0.0, 0.1, 0.2, 0.3]
    
    return analyze_static_noise_std(N, theta, deviation_ranges, steps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze static noise standard deviation vs time')
    parser.add_argument('--N', type=int, default=20, help='System size')
    parser.add_argument('--theta', type=float, default=np.pi/4, help='Theta parameter')
    parser.add_argument('--steps', type=int, default=10, help='Number of time steps')
    parser.add_argument('--deviations', nargs='+', type=float, default=[0.0, 0.1, 0.2, 0.3], 
                        help='List of deviations to analyze')
    parser.add_argument('--base-dir', default='experiments_data_samples_probDist', 
                        help='Base directory containing experiment results')
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_static_noise_std(
        N=args.N,
        theta=args.theta,
        deviation_ranges=args.deviations,
        steps=args.steps,
        base_dir=args.base_dir
    )
