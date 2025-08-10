#!/usr/bin/env python3
"""
Analyze static noise probability distributions.
This script loads mean probability distributions from static noise experiments,
calculates various metrics vs noise deviation, and creates comprehensive analysis plots.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from jaime_scripts import prob_distributions2std

def load_static_noise_probdist_data(base_dir="experiments_data_samples_probDist"):
    """
    Load mean probability distributions from static noise experiments.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment results
        
    Returns
    -------
    dict
        Dictionary with experiment data organized by deviation
    """
    static_noise_data = {}
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return static_noise_data
    
    # Look for static noise experiment directories
    static_noise_dirs = []
    for item in base_path.iterdir():
        if item.is_dir() and "static_noise_tesselation_static_noise" in item.name:
            static_noise_dirs.append(item)
    
    if not static_noise_dirs:
        print("No static noise experiment directories found!")
        return static_noise_data
    
    print(f"Found {len(static_noise_dirs)} static noise experiment directories:")
    for exp_dir in static_noise_dirs:
        print(f"  - {exp_dir.name}")
    
    # Process each static noise experiment directory
    for exp_dir in static_noise_dirs:
        print(f"\nProcessing {exp_dir.name}...")
        
        # Handle different directory structures:
        # 1. For nonoise: static_noise_tesselation_static_noise_nonoise/N_X/
        # 2. For noise: static_noise_tesselation_static_noise/dev_X.XXX/N_X/
        
        if "nonoise" in exp_dir.name:
            # Handle zero deviation case (no noise)
            deviation = 0.0
            dev_key = "dev_0.000"
            
            # Look directly for N_X subdirectory
            n_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("N_")]
            if not n_dirs:
                print(f"Warning: No N_X directory found in {exp_dir}")
                continue
            
            n_dir = n_dirs[0]  # Take the first N_X directory
            N = int(n_dir.name.split("_")[1])
            
            # Process this directory
            prob_distributions = []
            steps_found = 0
            
            # Find all mean_step_X.pkl files
            mean_files = sorted([f for f in n_dir.iterdir() if f.name.startswith("mean_step_") and f.name.endswith(".pkl")])
            
            for mean_file in mean_files:
                try:
                    with open(mean_file, "rb") as f:
                        mean_prob_dist = pickle.load(f)
                    prob_distributions.append(mean_prob_dist)
                    steps_found += 1
                except Exception as e:
                    print(f"Warning: Could not load {mean_file}: {e}")
                    prob_distributions.append(None)
            
            if steps_found > 0:
                static_noise_data[dev_key] = {
                    'deviation': deviation,
                    'N': N,
                    'steps': steps_found,
                    'prob_distributions': prob_distributions,
                    'exp_dir': str(n_dir)
                }
                print(f"  Loaded {dev_key}: {steps_found} steps, N={N}")
            else:
                print(f"  Warning: No valid probability distributions found for {dev_key}")
        
        else:
            # Handle noise cases with dev_X.XXX subdirectories
            for subdir in exp_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("dev_"):
                    try:
                        deviation = float(subdir.name.split("_")[1])
                        dev_key = subdir.name
                    except (IndexError, ValueError):
                        print(f"Warning: Could not parse deviation from {subdir.name}")
                        continue
                    
                    # Look for N_X subdirectory
                    n_dirs = [d for d in subdir.iterdir() if d.is_dir() and d.name.startswith("N_")]
                    if not n_dirs:
                        print(f"Warning: No N_X directory found in {subdir}")
                        continue
                    
                    n_dir = n_dirs[0]  # Take the first N_X directory
                    N = int(n_dir.name.split("_")[1])
                    
                    # Load mean probability distributions from this directory
                    prob_distributions = []
                    steps_found = 0
                    
                    # Find all mean_step_X.pkl files
                    mean_files = sorted([f for f in n_dir.iterdir() if f.name.startswith("mean_step_") and f.name.endswith(".pkl")])
                    
                    for mean_file in mean_files:
                        try:
                            with open(mean_file, "rb") as f:
                                mean_prob_dist = pickle.load(f)
                            prob_distributions.append(mean_prob_dist)
                            steps_found += 1
                        except Exception as e:
                            print(f"Warning: Could not load {mean_file}: {e}")
                            prob_distributions.append(None)
                    
                    if steps_found > 0:
                        static_noise_data[dev_key] = {
                            'deviation': deviation,
                            'N': N,
                            'steps': steps_found,
                            'prob_distributions': prob_distributions,
                            'exp_dir': str(n_dir)
                        }
                        print(f"  Loaded {dev_key}: {steps_found} steps, N={N}")
                    else:
                        print(f"  Warning: No valid probability distributions found for {dev_key}")
    
    return static_noise_data

def calculate_static_noise_metrics(static_noise_data):
    """
    Calculate various metrics for static noise analysis.
    
    Parameters
    ----------
    static_noise_data : dict
        Dictionary with static noise experiment data
        
    Returns
    -------
    dict
        Dictionary with calculated metrics
    """
    metrics = {
        'deviations': [],
        'max_probabilities': [],
        'max_prob_positions': [],
        'total_variations': [],
        'spreads': [],
        'entropies': [],
        'std_evolution': {},
        'prob_distributions_final': {}
    }
    
    # Sort by deviation
    sorted_devs = sorted(static_noise_data.keys(), key=lambda x: static_noise_data[x]['deviation'])
    
    for dev_key in sorted_devs:
        data = static_noise_data[dev_key]
        deviation = data['deviation']
        N = data['N']
        prob_distributions = data['prob_distributions']
        
        print(f"Calculating metrics for deviation {deviation:.3f}...")
        
        # Get the final probability distribution (last step with valid data)
        final_prob_dist = None
        for prob_dist in reversed(prob_distributions):
            if prob_dist is not None:
                final_prob_dist = prob_dist
                break
        
        if final_prob_dist is None:
            print(f"Warning: No valid final distribution for deviation {deviation:.3f}")
            continue
        
        # Store basic info
        metrics['deviations'].append(deviation)
        metrics['prob_distributions_final'][dev_key] = final_prob_dist
        
        # Calculate metrics for final distribution
        max_prob = np.max(final_prob_dist)
        max_pos = np.argmax(final_prob_dist)
        metrics['max_probabilities'].append(max_prob)
        metrics['max_prob_positions'].append(max_pos)
        
        # Total variation from uniform distribution
        uniform_prob = 1.0 / N
        total_var = np.sum(np.abs(final_prob_dist - uniform_prob))
        metrics['total_variations'].append(total_var)
        
        # Position spread (standard deviation)
        domain = np.arange(N) - N//2
        prob_normalized = final_prob_dist / np.sum(final_prob_dist)
        mean_position = np.sum(domain * prob_normalized.flatten())
        variance = np.sum(((domain - mean_position) ** 2) * prob_normalized.flatten())
        spread = np.sqrt(variance)
        metrics['spreads'].append(spread)
        
        # Shannon entropy
        prob_clean = final_prob_dist[final_prob_dist > 1e-12]
        entropy = -np.sum(prob_clean * np.log2(prob_clean))
        metrics['entropies'].append(entropy)
        
        # Calculate standard deviation evolution over time
        std_values = prob_distributions2std(prob_distributions, domain)
        metrics['std_evolution'][dev_key] = std_values
        
        print(f"  Max prob: {max_prob:.4f} at position {max_pos}")
        print(f"  Total variation: {total_var:.4f}")
        print(f"  Spread: {spread:.3f}")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Std evolution: {len(std_values)} time steps")
    
    return metrics

def plot_static_noise_analysis(metrics, static_noise_data, output_prefix="static_noise_analysis"):
    """
    Create comprehensive plots for static noise analysis.
    
    Parameters
    ----------
    metrics : dict
        Dictionary with calculated metrics
    static_noise_data : dict
        Original static noise data to get N values
    output_prefix : str
        Prefix for output filenames
    """
    
    # Group data by system size N
    n_groups = {}
    for dev_key, data in static_noise_data.items():
        N = data['N']
        if N not in n_groups:
            n_groups[N] = {'deviations': [], 'prob_dists': [], 'dev_keys': []}
        
        # Find the corresponding metrics index
        dev_value = data['deviation']
        for i, metric_dev in enumerate(metrics['deviations']):
            if abs(metric_dev - dev_value) < 1e-6:
                n_groups[N]['deviations'].append(dev_value)
                n_groups[N]['prob_dists'].append(list(metrics['prob_distributions_final'].values())[i])
                n_groups[N]['dev_keys'].append(dev_key)
                break
    
    # Create plots for each system size
    for N, group_data in n_groups.items():
        print(f"Creating plots for N={N} with {len(group_data['deviations'])} deviations...")
        
        # Create a comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot 1: Final probability distributions
        ax1 = plt.subplot(3, 3, 1)
        domain = np.arange(N) - N//2
        colors = plt.cm.viridis(np.linspace(0, 1, len(group_data['deviations'])))
        
        for i, (dev, prob_dist) in enumerate(zip(group_data['deviations'], group_data['prob_dists'])):
            ax1.plot(domain, prob_dist.flatten(), color=colors[i], 
                    label=f'σ={dev:.3f}', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Final Probability Distributions (N={N})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Maximum probability vs deviation
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(group_data['deviations'], [metrics['max_probabilities'][metrics['deviations'].index(d)] for d in group_data['deviations']], 'ro-', 
                 linewidth=3, markersize=8, markerfacecolor='red', markeredgecolor='darkred')
        ax2.set_xlabel('Noise Deviation (σ)')
        ax2.set_ylabel('Maximum Probability')
        ax2.set_title(f'Peak Probability vs Noise (N={N})')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Total variation vs deviation
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(group_data['deviations'], [metrics['total_variations'][metrics['deviations'].index(d)] for d in group_data['deviations']], 'bo-', 
                 linewidth=3, markersize=8, markerfacecolor='blue', markeredgecolor='darkblue')
        ax3.set_xlabel('Noise Deviation (σ)')
        ax3.set_ylabel('Total Variation from Uniform')
        ax3.set_title(f'Departure from Uniformity (N={N})')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Position spread vs deviation
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(group_data['deviations'], [metrics['spreads'][metrics['deviations'].index(d)] for d in group_data['deviations']], 'go-', 
                 linewidth=3, markersize=8, markerfacecolor='green', markeredgecolor='darkgreen')
        ax4.set_xlabel('Noise Deviation (σ)')
        ax4.set_ylabel('Position Spread (σ_pos)')
        ax4.set_title(f'Spatial Spreading vs Noise (N={N})')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Entropy vs deviation
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(group_data['deviations'], [metrics['entropies'][metrics['deviations'].index(d)] for d in group_data['deviations']], 'mo-', 
                 linewidth=3, markersize=8, markerfacecolor='magenta', markeredgecolor='darkmagenta')
        ax5.set_xlabel('Noise Deviation (σ)')
        ax5.set_ylabel('Shannon Entropy')
        ax5.set_title(f'Information Content vs Noise (N={N})')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Standard deviation evolution over time
        ax6 = plt.subplot(3, 3, 6)
        for i, dev_key in enumerate(group_data['dev_keys']):
            std_values = metrics['std_evolution'][dev_key]
            if std_values:
                time_steps = np.arange(len(std_values))
                ax6.plot(time_steps, std_values, color=colors[i], 
                        label=f'σ={group_data["deviations"][i]:.3f}', alpha=0.8, linewidth=2)
        
        ax6.set_xlabel('Time Steps')
        ax6.set_ylabel('Position Standard Deviation')
        ax6.set_title(f'Std Evolution Over Time (N={N})')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Max probability position vs deviation
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(group_data['deviations'], [metrics['max_prob_positions'][metrics['deviations'].index(d)] for d in group_data['deviations']], 'co-', 
                 linewidth=3, markersize=8, markerfacecolor='cyan', markeredgecolor='darkcyan')
        ax7.set_xlabel('Noise Deviation (σ)')
        ax7.set_ylabel('Position of Maximum Probability')
        ax7.set_title(f'Peak Position vs Noise (N={N})')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Probability distributions comparison (log scale)
        ax8 = plt.subplot(3, 3, 8)
        for i, (dev, prob_dist) in enumerate(zip(group_data['deviations'], group_data['prob_dists'])):
            ax8.semilogy(domain, prob_dist.flatten() + 1e-12, color=colors[i], 
                        label=f'σ={dev:.3f}', alpha=0.8, linewidth=2)
        
        ax8.set_xlabel('Position')
        ax8.set_ylabel('Probability (log scale)')
        ax8.set_title(f'Final Distributions (Log Scale, N={N})')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Summary statistics table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary table for this N group
        table_data = []
        for dev in group_data['deviations']:
            idx = metrics['deviations'].index(dev)
            row = [
                f'{dev:.3f}',
                f'{metrics["max_probabilities"][idx]:.4f}',
                f'{metrics["total_variations"][idx]:.3f}',
                f'{metrics["spreads"][idx]:.2f}',
                f'{metrics["entropies"][idx]:.2f}'
            ]
            table_data.append(row)
        
        table = ax9.table(cellText=table_data,
                         colLabels=['Deviation', 'Max Prob', 'Total Var', 'Spread', 'Entropy'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax9.set_title(f'Summary Statistics (N={N})', pad=20)
        
        plt.tight_layout()
        
        # Save the comprehensive plot
        plot_filename = f'{output_prefix}_N{N}_comprehensive.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Comprehensive analysis plot saved: {plot_filename}")
        
        plt.show()
        
        # Create individual focused plots for this N
        create_focused_plots_for_n(group_data, metrics, N, output_prefix)

def create_focused_plots_for_n(group_data, metrics, N, output_prefix):
    """Create individual focused plots for specific system size"""
def create_focused_plots_for_n(group_data, metrics, N, output_prefix):
    """Create individual focused plots for specific system size"""
    
    # Focused plot 1: Noise effects on distribution shape
    plt.figure(figsize=(12, 8))
    domain = np.arange(N) - N//2
    colors = plt.cm.viridis(np.linspace(0, 1, len(group_data['deviations'])))
    
    for i, (dev, prob_dist) in enumerate(zip(group_data['deviations'], group_data['prob_dists'])):
        plt.plot(domain, prob_dist.flatten(), color=colors[i], 
                label=f'σ={dev:.3f}', alpha=0.8, linewidth=2)
    
    plt.xlabel('Position', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.title(f'Static Noise Effects on Quantum Walk Distribution (N={N})', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_N{N}_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Focused plot 2: Key metrics vs noise deviation
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Max probability
    y_max_prob = [metrics['max_probabilities'][metrics['deviations'].index(d)] for d in group_data['deviations']]
    ax1.plot(group_data['deviations'], y_max_prob, 'ro-', linewidth=3, markersize=8)
    ax1.set_xlabel('Noise Deviation (σ)')
    ax1.set_ylabel('Maximum Probability')
    ax1.set_title(f'Peak Probability vs Noise (N={N})')
    ax1.grid(True, alpha=0.3)
    
    # Total variation
    y_total_var = [metrics['total_variations'][metrics['deviations'].index(d)] for d in group_data['deviations']]
    ax2.plot(group_data['deviations'], y_total_var, 'bo-', linewidth=3, markersize=8)
    ax2.set_xlabel('Noise Deviation (σ)')
    ax2.set_ylabel('Total Variation from Uniform')
    ax2.set_title(f'Departure from Uniformity (N={N})')
    ax2.grid(True, alpha=0.3)
    
    # Position spread
    y_spreads = [metrics['spreads'][metrics['deviations'].index(d)] for d in group_data['deviations']]
    ax3.plot(group_data['deviations'], y_spreads, 'go-', linewidth=3, markersize=8)
    ax3.set_xlabel('Noise Deviation (σ)')
    ax3.set_ylabel('Position Spread (σ_pos)')
    ax3.set_title(f'Spatial Spreading (N={N})')
    ax3.grid(True, alpha=0.3)
    
    # Entropy
    y_entropies = [metrics['entropies'][metrics['deviations'].index(d)] for d in group_data['deviations']]
    ax4.plot(group_data['deviations'], y_entropies, 'mo-', linewidth=3, markersize=8)
    ax4.set_xlabel('Noise Deviation (σ)')
    ax4.set_ylabel('Shannon Entropy')
    ax4.set_title(f'Information Content (N={N})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_N{N}_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_analysis_results(metrics, static_noise_data, output_filename="static_noise_analysis_results.pkl"):
    """Save complete analysis results"""
    
    results = {
        'metrics': metrics,
        'static_noise_data': static_noise_data,
        'analysis_info': {
            'num_deviations': len(metrics['deviations']),
            'deviation_range': [min(metrics['deviations']), max(metrics['deviations'])],
            'N': static_noise_data[list(static_noise_data.keys())[0]]['N'] if static_noise_data else None
        }
    }
    
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Analysis results saved to: {output_filename}")

def main(base_dir="experiments_data_samples_probDist", output_prefix="static_noise_analysis"):
    """
    Main analysis function for static noise experiments.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment results
    output_prefix : str
        Prefix for output files
    """
    print("=" * 80)
    print("STATIC NOISE PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Analyzing static noise experiments from: {base_dir}")
    
    # Load static noise data
    print("\n📂 Loading static noise experiment data...")
    static_noise_data = load_static_noise_probdist_data(base_dir)
    
    if not static_noise_data:
        print("❌ No static noise data found!")
        return None
    
    print(f"✅ Loaded data for {len(static_noise_data)} deviation levels")
    
    # Extract system parameters
    N_values = list(set(data['N'] for data in static_noise_data.values()))
    print(f"📊 System parameters: N = {N_values}")
    if len(N_values) > 1:
        print(f"   Note: Multiple system sizes detected - plots will be grouped by N")
    
    # Calculate metrics
    print("\n🔬 Calculating analysis metrics...")
    metrics = calculate_static_noise_metrics(static_noise_data)
    
    print(f"✅ Calculated metrics for {len(metrics['deviations'])} deviations")
    print(f"   Deviation range: {min(metrics['deviations']):.3f} to {max(metrics['deviations']):.3f}")
    
    # Create plots
    print("\n📈 Creating analysis plots...")
    plot_static_noise_analysis(metrics, static_noise_data, output_prefix)
    
    # Save results
    print("\n💾 Saving analysis results...")
    save_analysis_results(metrics, static_noise_data, f"{output_prefix}_results.pkl")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Processed {len(metrics['deviations'])} noise deviation levels:")
    for i, dev in enumerate(metrics['deviations']):
        print(f"  σ={dev:.3f}: max_prob={metrics['max_probabilities'][i]:.4f}, "
              f"spread={metrics['spreads'][i]:.2f}, entropy={metrics['entropies'][i]:.2f}")
    
    print(f"\nGenerated files:")
    print(f"  📈 {output_prefix}_comprehensive.png - Complete analysis plot")
    print(f"  📈 {output_prefix}_distributions.png - Probability distributions")
    print(f"  📈 {output_prefix}_metrics.png - Key metrics vs noise")
    print(f"  💾 {output_prefix}_results.pkl - Complete results data")
    print("=" * 80)
    
    return {
        'metrics': metrics,
        'static_noise_data': static_noise_data,
        'N_values': N_values
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze static noise probability distributions')
    parser.add_argument('--base-dir', default='experiments_data_samples_probDist', 
                        help='Base directory containing experiment results')
    parser.add_argument('--output-prefix', default='static_noise_analysis',
                        help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Run analysis
    results = main(
        base_dir=args.base_dir,
        output_prefix=args.output_prefix
    )
    
    if results:
        print(f"\n✅ Static noise analysis completed successfully!")
    else:
        print(f"\n❌ Static noise analysis failed!")
