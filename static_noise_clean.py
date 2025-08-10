#!/usr/bin/env python3
"""
Clean cluster-compatible static noise experiment using the cluster decorator module.
This version implements comprehensive static noise analysis for staggered quantum walks
with smart loading, automatic sample aggregation, and full result management.
"""

import time
import numpy as np
from cluster_module import cluster_deploy


# @cluster_deploy(
#     experiment_name="static_noise",
#     noise_type="static_noise",
#     N=200,  # Smaller default for local testing
#     samples=50
# )
def run_static_noise_experiment():
    """Run the static noise staggered quantum walk experiment with comprehensive features."""
    
    # Import after cluster environment is set up
    try:
        import numpy as np
        import networkx as nx
        import pickle
        import os
        import matplotlib.pyplot as plt
        
        # Import the static noise function
        from StaggeredQW_static_noise import staggered_qwalk_with_noise, cycle_tesselation_alpha, cycle_tesselation_beta
        
        # Import smart loading module functions
        from smart_loading import (
            smart_load_or_create_experiment,
            run_and_save_experiment_samples,
            create_mean_probability_distributions,
            load_mean_probability_distributions,
            check_mean_probability_distributions_exist,
            get_experiment_dir
        )
        
        # Import other shared functions from jaime_scripts
        from jaime_scripts import prob_distributions2std
        
        # Import states and experiments functions
        from sqw.states import uniform_initial_state
        
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        raise

    print("Starting static noise staggered quantum walk experiment...")
    
    # Experiment parameters
    N = 100  # System size (smaller for testing, increase for production)
    theta = np.pi / 3  # Base theta parameter
    steps = N//4  # Time steps
    samples = 10  # Samples per deviation
    init_nodes = [N//2]  # Start at center node
    
    # List of static noise deviations to test
    deviation_ranges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    print(f"Experiment parameters:")
    print(f"  N = {N}")
    print(f"  theta = π/{4 if theta == np.pi/4 else f'{np.pi/theta:.1f}'}")
    print(f"  steps = {steps}")
    print(f"  samples = {samples}")
    print(f"  init_nodes = {init_nodes}")
    print(f"  deviation_ranges = {deviation_ranges}")
    print()
    
    # Use smart loading functionality
    print("🧠 Using smart load system for efficient data management...")
    start_time = time.time()
    
    # Create tessellation function for static noise
    def static_noise_tesselation(N):
        """Static noise tessellation function wrapper"""
        # This serves as an identifier for the experiment type
        return nx.cycle_graph(N)  # Return a basic cycle graph as placeholder
    
    # Create graph function
    def static_noise_graph(N):
        """Graph function for static noise experiments"""
        return nx.cycle_graph(N)
    
    # Use the simplified smart loading approach
    try:
        # Create parameter lists for the smart loading system
        deviation_ranges_single = deviation_ranges  # List of deviations
        
        # Use a simpler approach - run each deviation separately using smart loading
        all_mean_results = []
        
        for dev_idx, dev in enumerate(deviation_ranges):
            print(f"\n--- Processing deviation {dev:.3f} ({dev_idx+1}/{len(deviation_ranges)}) ---")
            
            # Create angles list for this deviation only
            dev_angles_list = []
            for sample in range(samples):
                dev_angles_list.append([[theta, theta]] * steps)
            
            # Use smart loading for this single deviation
            single_dev_result = smart_load_or_create_experiment(
                graph_func=static_noise_graph,
                tesselation_func=static_noise_tesselation,
                N=N,
                steps=steps,
                angles_or_angles_list=dev_angles_list,
                tesselation_order_or_list=[[0, 1] for x in range(steps)],
                initial_state_func=uniform_initial_state,
                initial_state_kwargs={"nodes": init_nodes},
                parameter_list=[dev],  # Single deviation
                samples=samples,
                noise_type="static_noise",
                parameter_name="dev",
                samples_base_dir="experiments_data_samples",
                probdist_base_dir="experiments_data_samples_probDist"
            )
            
            # Extract the final step result (static noise only gives final state)
            if single_dev_result and len(single_dev_result) > 0 and len(single_dev_result[0]) > 0:
                # Get the last step result
                final_step_result = single_dev_result[0][-1]  # [dev][step] -> final step
                all_mean_results.append(final_step_result)
            else:
                print(f"⚠️  No results for deviation {dev:.3f}, using fallback")
                fallback_result = run_single_static_noise_sample(N, theta, steps, init_nodes, dev)
                all_mean_results.append(fallback_result['probabilities'])
        
        mean_results = all_mean_results
        print("✅ Smart loading completed successfully!")
        
    except Exception as e:
        print(f"⚠️  Smart loading failed: {e}")
        print("📝 Falling back to direct experiment execution...")
        import traceback
        traceback.print_exc()
        mean_results = run_direct_static_noise_experiment(
            N, theta, steps, init_nodes, deviation_ranges, samples
        )
    
    experiment_time = time.time() - start_time
    print(f"\n🎉 Static noise experiment completed in {experiment_time:.2f} seconds")
    
    # Analysis and visualization
    print("\n📊 Analyzing static noise effects...")
    try:
        analysis_results = analyze_static_noise_results(
            mean_results, deviation_ranges, N, steps
        )
        
        # Create comprehensive plots
        create_static_noise_plots(
            analysis_results, deviation_ranges, N, theta, steps, samples
        )
        
        # Save detailed results
        save_static_noise_results(
            analysis_results, deviation_ranges, N, theta, steps, samples
        )
        
        print("✅ Analysis and visualization completed!")
        
    except Exception as e:
        print(f"⚠️  Analysis failed: {e}")
        print("Results are still available in the experiment directories.")
    
    # Performance summary
    print("\n=== Performance Summary ===")
    print(f"System size (N): {N}")
    print(f"Base theta: {theta:.4f}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(deviation_ranges)}")
    print(f"Total quantum walks: {len(deviation_ranges) * samples}")
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(deviation_ranges) * samples):.3f} seconds")
    
    print("\n=== Static Noise Model Details ===")
    print("Static noise model:")
    print("- deviation=0.0: Perfect tessellation (no noise)")
    print("- deviation>0: Random deviations applied to tessellation edges")
    print("- Each sample generates different random noise parameters")
    print("- Red and blue tessellation edges get independent noise")
    print("- Mean probability distributions average over all samples")
    
    return {
        "deviation_ranges": deviation_ranges,
        "N": N,
        "theta": theta,
        "steps": steps,
        "samples": samples,
        "total_time": experiment_time,
        "analysis_results": analysis_results if 'analysis_results' in locals() else None
    }


def smart_load_static_noise_experiment(N, theta, steps, init_nodes, deviation_ranges, samples):
    """
    Smart loading system for static noise experiments
    """
    import os
    import pickle
    
    base_dir = "experiments_data_static_noise"
    probdist_dir = "experiments_data_static_noise_probDist"
    
    # Step 1: Try to load mean probability distributions
    print("Step 1: Checking for existing mean probability distributions...")
    
    probdist_exists = True
    for dev in deviation_ranges:
        exp_dir = get_static_noise_exp_dir(N, theta, dev, base_dir=probdist_dir)
        for step_idx in range(steps):
            prob_file = os.path.join(exp_dir, f"mean_prob_step_{step_idx}.pkl")
            if not os.path.exists(prob_file):
                probdist_exists = False
                break
        if not probdist_exists:
            break
    
    if probdist_exists:
        print("✅ Found existing mean probability distributions - loading directly!")
        return load_static_noise_probdist(N, theta, steps, deviation_ranges, probdist_dir)
    
    # Step 2: Try to load samples and create probability distributions
    print("Step 2: Checking for existing sample data...")
    
    samples_exist = True
    for dev in deviation_ranges:
        exp_dir = get_static_noise_exp_dir(N, theta, dev, base_dir=base_dir)
        for sample_idx in range(samples):
            sample_file = os.path.join(exp_dir, f"sample_{sample_idx}.pkl")
            if not os.path.exists(sample_file):
                samples_exist = False
                break
        if not samples_exist:
            break
    
    if samples_exist:
        print("✅ Found existing samples - creating mean probability distributions...")
        create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples, base_dir, probdist_dir)
        return load_static_noise_probdist(N, theta, steps, deviation_ranges, probdist_dir)
    
    # Step 3: Run new experiment
    print("Step 3: No existing data found - running new experiment...")
    return run_direct_static_noise_experiment(N, theta, steps, init_nodes, deviation_ranges, samples)


def run_direct_static_noise_experiment(N, theta, steps, init_nodes, deviation_ranges, samples):
    """
    Run the static noise experiment directly with immediate saving
    """
    import os
    import pickle
    from tqdm import tqdm
    
    base_dir = "experiments_data_static_noise"
    probdist_dir = "experiments_data_static_noise_probDist"
    
    print(f"🔄 Running {samples} samples for each of {len(deviation_ranges)} deviations...")
    
    total_samples = len(deviation_ranges) * samples
    completed_samples = 0
    
    # Run experiments with immediate saving
    for dev_idx, deviation in enumerate(tqdm(deviation_ranges, desc="Deviations")):
        print(f"\n=== Processing deviation {deviation:.3f} ({dev_idx+1}/{len(deviation_ranges)}) ===")
        
        exp_dir = get_static_noise_exp_dir(N, theta, deviation, base_dir=base_dir)
        os.makedirs(exp_dir, exist_ok=True)
        
        dev_samples = []
        for sample_idx in tqdm(range(samples), desc=f"Samples (dev={deviation:.3f})", leave=False):
            sample_file = os.path.join(exp_dir, f"sample_{sample_idx}.pkl")
            
            if os.path.exists(sample_file):
                # Load existing sample
                with open(sample_file, 'rb') as f:
                    sample_result = pickle.load(f)
                dev_samples.append(sample_result)
            else:
                # Run new sample
                sample_result = run_single_static_noise_sample(N, theta, steps, init_nodes, deviation)
                
                # Save sample immediately
                with open(sample_file, 'wb') as f:
                    pickle.dump(sample_result, f)
                
                dev_samples.append(sample_result)
            
            completed_samples += 1
        
        print(f"✅ Deviation {deviation:.3f} completed: {len(dev_samples)} samples")
    
    print(f"🎉 All samples completed: {completed_samples}/{total_samples}")
    
    # Create mean probability distributions
    print("📊 Creating mean probability distributions...")
    create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples, base_dir, probdist_dir)
    
    # Load and return results
    return load_static_noise_probdist(N, theta, steps, deviation_ranges, probdist_dir)


def run_single_static_noise_sample(N, theta, steps, init_nodes, deviation):
    """
    Run a single static noise sample
    """
    from StaggeredQW_static_noise import staggered_qwalk_with_noise
    
    # Run the quantum walk with static noise
    probabilities, Hr_clean, Hb_clean, Hr_noisy, Hb_noisy, red_noise, blue_noise = \
        staggered_qwalk_with_noise(N, theta, steps, init_nodes, deviation)
    
    return {
        'probabilities': probabilities,
        'Hr_clean': Hr_clean,
        'Hb_clean': Hb_clean,
        'Hr_noisy': Hr_noisy,
        'Hb_noisy': Hb_noisy,
        'red_noise': red_noise,
        'blue_noise': blue_noise,
        'deviation': deviation,
        'N': N,
        'theta': theta,
        'steps': steps,
        'init_nodes': init_nodes
    }


def get_static_noise_exp_dir(N, theta, deviation, base_dir="experiments_data_static_noise"):
    """
    Get experiment directory for static noise experiments
    """
    import os
    theta_str = f"theta_{theta:.4f}".replace('.', 'p')
    dev_str = f"dev_{deviation:.3f}".replace('.', 'p')
    return os.path.join(base_dir, f"static_noise_N{N}_{theta_str}_{dev_str}")


def create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples, base_dir, probdist_dir):
    """
    Create mean probability distributions from saved samples
    """
    import os
    import pickle
    import numpy as np
    
    print("Creating mean probability distributions from samples...")
    
    for deviation in deviation_ranges:
        print(f"  Processing deviation {deviation:.3f}...")
        
        # Load all samples for this deviation
        exp_dir = get_static_noise_exp_dir(N, theta, deviation, base_dir=base_dir)
        sample_probs = []
        
        for sample_idx in range(samples):
            sample_file = os.path.join(exp_dir, f"sample_{sample_idx}.pkl")
            with open(sample_file, 'rb') as f:
                sample_result = pickle.load(f)
                sample_probs.append(sample_result['probabilities'])
        
        # Calculate mean probability for this step
        prob_array = np.array(sample_probs).squeeze()  # Shape: (samples, N)
        mean_prob = np.mean(prob_array, axis=0)
        
        # Save mean probability
        probdist_exp_dir = get_static_noise_exp_dir(N, theta, deviation, base_dir=probdist_dir)
        os.makedirs(probdist_exp_dir, exist_ok=True)
        
        # For static noise, we only have one final step, but we create the file structure
        # to match the expected format
        prob_file = os.path.join(probdist_exp_dir, f"mean_prob_step_{steps-1}.pkl")
        with open(prob_file, 'wb') as f:
            pickle.dump(mean_prob, f)
    
    print("✅ Mean probability distributions created!")


def load_static_noise_probdist(N, theta, steps, deviation_ranges, probdist_dir):
    """
    Load mean probability distributions for static noise experiments
    """
    import os
    import pickle
    
    results = []
    for deviation in deviation_ranges:
        exp_dir = get_static_noise_exp_dir(N, theta, deviation, base_dir=probdist_dir)
        prob_file = os.path.join(exp_dir, f"mean_prob_step_{steps-1}.pkl")
        
        with open(prob_file, 'rb') as f:
            mean_prob = pickle.load(f)
            results.append(mean_prob)
    
    return results


def analyze_static_noise_results(mean_results, deviation_ranges, N, steps):
    """
    Analyze static noise experiment results
    """
    import numpy as np
    
    print("Analyzing static noise effects...")
    
    # Calculate statistics for each deviation
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
    domain = np.arange(N) - N//2  # Center domain around 0
    
    for i, (dev, mean_prob) in enumerate(zip(deviation_ranges, mean_results)):
        # Maximum probability and its node
        max_prob = np.max(mean_prob)
        max_node = np.argmax(mean_prob)
        analysis['max_probabilities'].append(max_prob)
        analysis['max_prob_nodes'].append(max_node)
        
        # Total variation from uniform distribution
        total_var = np.sum(np.abs(mean_prob - uniform_prob))
        analysis['total_variations'].append(total_var)
        
        # Spread (standard deviation of position)
        prob_normalized = mean_prob / np.sum(mean_prob)  # Normalize just in case
        mean_position = np.sum(domain * prob_normalized.flatten())
        variance = np.sum(((domain - mean_position) ** 2) * prob_normalized.flatten())
        spread = np.sqrt(variance)
        analysis['spreads'].append(spread)
        
        # Shannon entropy
        prob_clean = mean_prob[mean_prob > 1e-12]  # Avoid log(0)
        entropy = -np.sum(prob_clean * np.log2(prob_clean))
        analysis['entropy'].append(entropy)
        
        print(f"Deviation {dev:.3f}: max_prob={max_prob:.4f} at node {max_node}, "
              f"total_var={total_var:.4f}, spread={spread:.3f}, entropy={entropy:.3f}")
    
    return analysis


def create_static_noise_plots(analysis_results, deviation_ranges, N, theta, steps, samples):
    """
    Create comprehensive plots for static noise analysis
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Creating static noise analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean probabilities vs position for each deviation
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
    
    # Save the plot
    plot_filename = f"static_noise_analysis_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Analysis plots saved to {plot_filename}")
    
    plt.show()


def save_static_noise_results(analysis_results, deviation_ranges, N, theta, steps, samples):
    """
    Save detailed results to files
    """
    import pickle
    import pandas as pd
    import os
    
    # Create results directory
    results_dir = "static_noise_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save complete analysis as pickle
    analysis_filename = os.path.join(results_dir, f"static_noise_analysis_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.pkl")
    with open(analysis_filename, 'wb') as f:
        pickle.dump(analysis_results, f)
    
    # Save summary as CSV
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


# Example usage and testing
if __name__ == "__main__":
    print("Static Noise Staggered Quantum Walk - Comprehensive Analysis")
    print("=" * 65)
    
    # For testing, you can run this directly without cluster deployment
    try:
        result = run_static_noise_experiment()
        print("\n✅ Experiment completed successfully!")
        print(f"Results: {result}")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
