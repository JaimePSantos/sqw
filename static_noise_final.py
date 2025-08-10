#!/usr/bin/env python3
"""
Static Noise Clean - Simplified Version with Smart Loading Integration
This version properly integrates with your existing smart loading infrastructure
"""

import time
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import modules
from StaggeredQW_static_noise import staggered_qwalk_with_noise
from smart_loading import get_experiment_dir, create_mean_probability_distributions, load_mean_probability_distributions, check_mean_probability_distributions_exist
from sqw.states import uniform_initial_state, amp2prob
from jaime_scripts import prob_distributions2std
from cluster_module import cluster_deploy


@cluster_deploy(
    experiment_name="static_noise",
    noise_type="static_noise",
    N=200,
    samples=50
)
def run_static_noise_experiment():
    """Run static noise experiment with proper smart loading integration"""
    
    print("Starting static noise staggered quantum walk experiment...")
    
    # Experiment parameters
    N = 200
    theta = np.pi / 4
    steps = 20
    samples = 50
    init_nodes = [N//2]
    deviation_ranges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    print(f"Experiment parameters:")
    print(f"  N = {N}")
    print(f"  theta = π/4")
    print(f"  steps = {steps}")
    print(f"  samples = {samples}")
    print(f"  init_nodes = {init_nodes}")
    print(f"  deviation_ranges = {deviation_ranges}")
    print()
    
    # Create tessellation function for directory naming
    def static_noise_tesselation(N):
        """Tessellation identifier for static noise"""
        return None
    
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    start_time = time.time()
    
    # Use smart loading hierarchy
    print("🧠 Using smart load system for efficient data management...")
    
    # Step 1: Try to load mean probability distributions
    print("Step 1: Checking for existing mean probability distributions...")
    if check_mean_probability_distributions_exist(
        static_noise_tesselation, N, steps, deviation_ranges,
        "experiments_data_samples_probDist", "static_noise"
    ):
        print("✅ Found existing mean probability distributions - loading directly!")
        mean_results = load_mean_probability_distributions(
            static_noise_tesselation, N, steps, deviation_ranges,
            "experiments_data_samples_probDist", "static_noise"
        )
        # Extract final step results for static noise
        final_results = [result[-1] if result else None for result in mean_results]
    else:
        # Step 2: Check for samples and create probability distributions  
        print("Step 2: Checking for existing sample data...")
        samples_exist = check_all_samples_exist(N, theta, deviation_ranges, samples, steps)
        
        if samples_exist:
            print("✅ Found existing samples - creating mean probability distributions...")
            create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples)
            mean_results = load_mean_probability_distributions(
                static_noise_tesselation, N, steps, deviation_ranges,
                "experiments_data_samples_probDist", "static_noise"
            )
            final_results = [result[-1] if result else None for result in mean_results]
        else:
            # Step 3: Run new experiment
            print("Step 3: No existing data found - running new experiment...")
            final_results = run_static_noise_samples_experiment(
                N, theta, steps, init_nodes, deviation_ranges, samples
            )
    
    experiment_time = time.time() - start_time
    print(f"\n🎉 Static noise experiment completed in {experiment_time:.2f} seconds")
    
    # Analysis and visualization
    print("\n📊 Analyzing static noise effects...")
    try:
        analysis_results = analyze_static_noise_results(final_results, deviation_ranges, N, steps)
        create_static_noise_plots(analysis_results, deviation_ranges, N, theta, steps, samples)
        save_static_noise_results(analysis_results, deviation_ranges, N, theta, steps, samples)
        print("✅ Analysis and visualization completed!")
        
    except Exception as e:
        print(f"⚠️  Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    return {
        "deviation_ranges": deviation_ranges,
        "N": N, "theta": theta, "steps": steps, "samples": samples,
        "total_time": experiment_time,
        "final_results": final_results
    }


def check_all_samples_exist(N, theta, deviation_ranges, samples, steps):
    """Check if all sample files exist"""
    
    def static_noise_tesselation(N):
        return None
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    for dev in deviation_ranges:
        has_noise = dev > 0
        noise_params = [dev]
        exp_dir = get_experiment_dir(
            static_noise_tesselation, has_noise, N, noise_params, "static_noise", "experiments_data_samples"
        )
        
        for sample_idx in range(samples):
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    return False
    return True


def run_static_noise_samples_experiment(N, theta, steps, init_nodes, deviation_ranges, samples):
    """Run the static noise experiment with immediate sample saving"""
    print(f"🔄 Running {samples} samples for each of {len(deviation_ranges)} deviations...")
    
    def static_noise_tesselation(N):
        return None
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    final_results = []
    
    for dev_idx, deviation in enumerate(tqdm(deviation_ranges, desc="Deviations")):
        print(f"\n=== Processing deviation {deviation:.3f} ({dev_idx+1}/{len(deviation_ranges)}) ===")
        
        # Create experiment directory following standard pattern
        has_noise = deviation > 0
        noise_params = [deviation]
        exp_dir = get_experiment_dir(
            static_noise_tesselation, has_noise, N, noise_params, "static_noise", "experiments_data_samples"
        )
        
        dev_samples = []
        for sample_idx in tqdm(range(samples), desc=f"Samples (dev={deviation:.3f})", leave=False):
            # Check if sample already exists
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                # Load existing sample (just load final step)
                final_step_dir = os.path.join(exp_dir, f"step_{steps-1}")
                final_filename = f"final_step_{steps-1}_sample{sample_idx}.pkl"
                final_filepath = os.path.join(final_step_dir, final_filename)
                with open(final_filepath, 'rb') as f:
                    sample_result = pickle.load(f)
                dev_samples.append(sample_result)
            else:
                # Run new sample
                probabilities, Hr_clean, Hb_clean, Hr_noisy, Hb_noisy, red_noise, blue_noise = \
                    staggered_qwalk_with_noise(N, theta, steps, init_nodes, deviation)
                
                # Convert to probability distribution
                prob_dist = amp2prob(probabilities)
                
                # Save sample in step-wise structure for compatibility
                for step_idx in range(steps):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    os.makedirs(step_dir, exist_ok=True)
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    
                    # For static noise, the result is the same for all steps since 
                    # it's a single-step calculation returning the final distribution
                    with open(filepath, 'wb') as f:
                        pickle.dump(prob_dist, f)
                
                dev_samples.append(prob_dist)
        
        # Calculate mean for this deviation
        mean_prob = np.mean(dev_samples, axis=0)
        final_results.append(mean_prob)
        
        print(f"✅ Deviation {deviation:.3f} completed: {len(dev_samples)} samples")
    
    print(f"🎉 All samples completed!")
    
    # Create mean probability distributions using the standard system
    print("📊 Creating mean probability distributions...")
    create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples)
    
    return final_results


def create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples):
    """Create mean probability distributions from samples using standard format"""
    # Create tessellation function
    def static_noise_tesselation(N):
        return None
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    # Use the standard function
    create_mean_probability_distributions(
        static_noise_tesselation, N, steps, deviation_ranges, samples,
        "experiments_data_samples", "experiments_data_samples_probDist", "static_noise"
    )


def analyze_static_noise_results(mean_results, deviation_ranges, N, steps):
    """Analyze static noise experiment results"""
    
    print("Analyzing static noise effects...")
    
    analysis = {
        'deviations': deviation_ranges,
        'mean_probabilities': mean_results,
        'max_probabilities': [],
        'max_prob_nodes': [],
        'total_variations': [],
        'spreads': [],
        'entropy': []
    }
    
    uniform_prob = 1.0 / N
    domain = np.arange(N) - N//2
    
    for i, (dev, mean_prob) in enumerate(zip(deviation_ranges, mean_results)):
        max_prob = np.max(mean_prob)
        max_node = np.argmax(mean_prob)
        analysis['max_probabilities'].append(max_prob)
        analysis['max_prob_nodes'].append(max_node)
        
        total_var = np.sum(np.abs(mean_prob - uniform_prob))
        analysis['total_variations'].append(total_var)
        
        prob_normalized = mean_prob / np.sum(mean_prob)
        mean_position = np.sum(domain * prob_normalized.flatten())
        variance = np.sum(((domain - mean_position) ** 2) * prob_normalized.flatten())
        spread = np.sqrt(variance)
        analysis['spreads'].append(spread)
        
        prob_clean = mean_prob[mean_prob > 1e-12]
        entropy = -np.sum(prob_clean * np.log2(prob_clean))
        analysis['entropy'].append(entropy)
        
        print(f"Deviation {dev:.3f}: max_prob={max_prob:.4f} at node {max_node}, "
              f"total_var={total_var:.4f}, spread={spread:.3f}, entropy={entropy:.3f}")
    
    return analysis


def create_static_noise_plots(analysis_results, deviation_ranges, N, theta, steps, samples):
    """Create comprehensive plots for static noise analysis"""
    
    print("Creating static noise analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean probabilities vs position
    domain = np.arange(N) - N//2
    colors = plt.cm.viridis(np.linspace(0, 1, len(deviation_ranges)))
    
    for i, (dev, mean_prob) in enumerate(zip(deviation_ranges, analysis_results['mean_probabilities'])):
        ax1.plot(domain, mean_prob.flatten(), color=colors[i], label=f'dev={dev:.3f}', alpha=0.8)
    
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Mean Probability')
    ax1.set_title('Mean Probability Distributions vs Position')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maximum probability vs deviation
    ax2.plot(deviation_ranges, analysis_results['max_probabilities'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Noise Deviation')
    ax2.set_ylabel('Maximum Probability')
    ax2.set_title('Maximum Probability vs Noise Deviation')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total variation vs deviation
    ax3.plot(deviation_ranges, analysis_results['total_variations'], 'bo-', linewidth=2, markersize=8)
    ax3.set_xlabel('Noise Deviation')
    ax3.set_ylabel('Total Variation from Uniform')
    ax3.set_title('Total Variation vs Noise Deviation')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Spread vs deviation
    ax4.plot(deviation_ranges, analysis_results['spreads'], 'go-', linewidth=2, markersize=8)
    ax4.set_xlabel('Noise Deviation')
    ax4.set_ylabel('Position Spread (σ)')
    ax4.set_title('Position Spread vs Noise Deviation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = f"static_noise_analysis_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to {plot_filename}")
    
    plt.show()


def save_static_noise_results(analysis_results, deviation_ranges, N, theta, steps, samples):
    """Save detailed results to files"""
    
    results_dir = "static_noise_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save complete analysis as pickle
    analysis_filename = os.path.join(results_dir, f"static_noise_analysis_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.pkl")
    with open(analysis_filename, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Save summary as CSV
    try:
        import pandas as pd
        summary_data = {
            'deviation': deviation_ranges,
            'max_probability': analysis_results['max_probabilities'],
            'max_prob_node': analysis_results['max_prob_nodes'],
            'total_variation': analysis_results['total_variations'],
            'position_spread': analysis_results['spreads'],
            'entropy': analysis_results['entropy']
        }
        
        df = pd.DataFrame(summary_data)
        csv_filename = os.path.join(results_dir, f"static_noise_summary_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.csv")
        df.to_csv(csv_filename, index=False)
        
        print(f"Results saved to:")
        print(f"  - Analysis: {analysis_filename}")
        print(f"  - Summary: {csv_filename}")
        
    except ImportError:
        print(f"Results saved to:")
        print(f"  - Analysis: {analysis_filename}")
        print("  - CSV summary skipped (pandas not available)")


if __name__ == "__main__":
    print("Static Noise Staggered Quantum Walk - Smart Loading Version")
    print("=" * 60)
    
    try:
        result = run_static_noise_experiment()
        print("\n✅ Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
