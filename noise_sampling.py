"""
Noise Sampling Module for Staggered Quantum Walk

This module provides functions to run multiple samples of the staggered quantum walk
with different deviation ranges to analyze the effects of static noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from StaggeredQW_static_noise import staggered_qwalk_with_noise
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def run_noise_samples(N, theta, steps, init_nodes, deviation_ranges, n_samples=100, 
                     parallel=True, n_processes=None):
    """
    Run multiple samples of staggered quantum walk with different noise deviations
    
    Parameters:
    - N: number of nodes
    - theta: base theta parameter
    - steps: number of evolution steps
    - init_nodes: list of initial nodes (empty list = uniform superposition)
    - deviation_ranges: list of deviation values to test
    - n_samples: number of samples to run for each deviation
    - parallel: whether to use parallel processing
    - n_processes: number of processes to use (None = auto)
    
    Returns:
    - results_dict: dictionary containing results for each deviation
    """
    print(f"Running {n_samples} samples for each of {len(deviation_ranges)} deviation values...")
    print(f"Total simulations: {n_samples * len(deviation_ranges)}")
    
    results_dict = {}
    
    for deviation in tqdm(deviation_ranges, desc="Processing deviations"):
        print(f"\nProcessing deviation = {deviation:.3f}")
        
        if parallel and n_samples > 1:
            # Use parallel processing
            if n_processes is None:
                n_processes = min(mp.cpu_count(), n_samples)
            
            # Create partial function with fixed parameters
            worker_func = partial(
                _single_sample_worker,
                N=N, theta=theta, steps=steps, 
                init_nodes=init_nodes, deviation_range=deviation
            )
            
            with mp.Pool(processes=n_processes) as pool:
                sample_results = list(tqdm(
                    pool.imap(worker_func, range(n_samples)),
                    total=n_samples,
                    desc=f"Samples (dev={deviation:.3f})",
                    leave=False
                ))
        else:
            # Sequential processing
            sample_results = []
            for sample_idx in tqdm(range(n_samples), 
                                 desc=f"Samples (dev={deviation:.3f})", 
                                 leave=False):
                result = _single_sample_worker(
                    sample_idx, N, theta, steps, init_nodes, deviation
                )
                sample_results.append(result)
        
        # Process results for this deviation
        probabilities_list = [result['probabilities'] for result in sample_results]
        red_noise_list = [result['red_noise'] for result in sample_results]
        blue_noise_list = [result['blue_noise'] for result in sample_results]
        
        # Calculate statistics
        prob_array = np.array(probabilities_list).squeeze()  # Shape: (n_samples, N)
        mean_probs = np.mean(prob_array, axis=0)
        std_probs = np.std(prob_array, axis=0)
        
        results_dict[deviation] = {
            'probabilities_all_samples': prob_array,
            'mean_probabilities': mean_probs,
            'std_probabilities': std_probs,
            'red_noise_samples': red_noise_list,
            'blue_noise_samples': blue_noise_list,
            'n_samples': n_samples
        }
    
    return results_dict


def _single_sample_worker(sample_idx, N, theta, steps, init_nodes, deviation_range):
    """
    Worker function for a single sample (used by multiprocessing)
    """
    probabilities, Hr_clean, Hb_clean, Hr_noisy, Hb_noisy, red_noise, blue_noise = \
        staggered_qwalk_with_noise(N, theta, steps, init_nodes, deviation_range)
    
    return {
        'sample_idx': sample_idx,
        'probabilities': probabilities,
        'red_noise': red_noise,
        'blue_noise': blue_noise
    }


def analyze_noise_effects(results_dict, save_plots=True, output_dir="./"):
    """
    Analyze the effects of noise on the quantum walk
    
    Parameters:
    - results_dict: results from run_noise_samples
    - save_plots: whether to save plots to files
    - output_dir: directory to save plots
    
    Returns:
    - analysis_dict: dictionary with analysis results
    """
    deviations = sorted(results_dict.keys())
    n_nodes = len(results_dict[deviations[0]]['mean_probabilities'])
    
    # Prepare data for analysis
    mean_probs_matrix = np.array([results_dict[dev]['mean_probabilities'] for dev in deviations])
    std_probs_matrix = np.array([results_dict[dev]['std_probabilities'] for dev in deviations])
    
    # Calculate total variation (measure of spread)
    total_variations = []
    max_prob_nodes = []
    
    for dev in deviations:
        mean_probs = results_dict[dev]['mean_probabilities']
        # Total variation: sum of absolute differences from uniform distribution
        uniform_prob = 1.0 / n_nodes
        total_var = np.sum(np.abs(mean_probs - uniform_prob))
        total_variations.append(total_var)
        
        # Node with maximum probability
        max_prob_nodes.append(np.argmax(mean_probs))
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean probabilities vs deviation
    for node in range(n_nodes):
        ax1.plot(deviations, mean_probs_matrix[:, node], 'o-', label=f'Node {node}')
    ax1.set_xlabel('Noise Deviation')
    ax1.set_ylabel('Mean Probability')
    ax1.set_title('Mean Probabilities vs Noise Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation vs deviation
    for node in range(n_nodes):
        ax2.plot(deviations, std_probs_matrix[:, node], 's-', label=f'Node {node}')
    ax2.set_xlabel('Noise Deviation')
    ax2.set_ylabel('Standard Deviation of Probability')
    ax2.set_title('Probability Std Dev vs Noise Deviation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total variation vs deviation
    ax3.plot(deviations, total_variations, 'ro-', linewidth=2, markersize=8)
    ax3.set_xlabel('Noise Deviation')
    ax3.set_ylabel('Total Variation from Uniform')
    ax3.set_title('Total Variation vs Noise Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Heatmap of mean probabilities
    im = ax4.imshow(mean_probs_matrix.T, aspect='auto', cmap='viridis', origin='lower')
    ax4.set_xlabel('Deviation Index')
    ax4.set_ylabel('Node')
    ax4.set_title('Mean Probabilities Heatmap')
    ax4.set_xticks(range(len(deviations)))
    ax4.set_xticklabels([f'{dev:.3f}' for dev in deviations], rotation=45)
    ax4.set_yticks(range(n_nodes))
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"{output_dir}/noise_analysis.png", dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved to {output_dir}/noise_analysis.png")
    
    plt.show()
    
    # Create summary statistics
    analysis_dict = {
        'deviations': deviations,
        'mean_probabilities_matrix': mean_probs_matrix,
        'std_probabilities_matrix': std_probs_matrix,
        'total_variations': total_variations,
        'max_prob_nodes': max_prob_nodes
    }
    
    return analysis_dict


def save_results_to_csv(results_dict, filename="noise_sampling_results.csv"):
    """
    Save results to CSV file for further analysis
    
    Parameters:
    - results_dict: results from run_noise_samples
    - filename: output CSV filename
    """
    data_rows = []
    
    for deviation, data in results_dict.items():
        n_samples = data['n_samples']
        n_nodes = len(data['mean_probabilities'])
        
        for sample_idx in range(n_samples):
            row = {
                'deviation': deviation,
                'sample': sample_idx,
            }
            
            # Add probability for each node
            for node in range(n_nodes):
                row[f'prob_node_{node}'] = data['probabilities_all_samples'][sample_idx, node]
            
            # Add noise parameters
            row['red_noise_params'] = str(data['red_noise_samples'][sample_idx])
            row['blue_noise_params'] = str(data['blue_noise_samples'][sample_idx])
            
            data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":
    print("Noise Sampling Analysis for Staggered Quantum Walk")
    print("=" * 55)
    
    # Parameters
    N = 6
    theta = np.pi / 4
    steps = 1
    init_nodes = [0]  # Start at node 0
    deviation_ranges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    n_samples = 50
    
    print(f"Parameters:")
    print(f"  N = {N}")
    print(f"  theta = π/{4 if theta == np.pi/4 else f'{np.pi/theta:.1f}'}")
    print(f"  steps = {steps}")
    print(f"  init_nodes = {init_nodes}")
    print(f"  deviation_ranges = {deviation_ranges}")
    print(f"  n_samples = {n_samples}")
    print()
    
    # Run the sampling
    results = run_noise_samples(
        N=N, 
        theta=theta, 
        steps=steps, 
        init_nodes=init_nodes,
        deviation_ranges=deviation_ranges,
        n_samples=n_samples,
        parallel=True
    )
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_noise_effects(results, save_plots=True)
    
    # Save to CSV
    save_results_to_csv(results, "noise_sampling_results.csv")
    
    # Print summary
    print("\nSummary:")
    print("=" * 20)
    for i, dev in enumerate(analysis['deviations']):
        print(f"Deviation {dev:.3f}: Total Variation = {analysis['total_variations'][i]:.6f}, "
              f"Max Prob Node = {analysis['max_prob_nodes'][i]}")
