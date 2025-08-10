#!/usr/bin/env python3
"""
Static Noise Production - Ready for cluster deployment
This is the production version with full parameters for your static noise experiments
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
    experiment_name="static_noise_production",
    noise_type="static_noise",
    N=200,
    samples=50
)
def run_static_noise_production_experiment():
    """
    Production static noise experiment with proper smart loading integration
    
    This function runs the full static noise experiment with:
    - N = 200 nodes
    - 50 samples per deviation
    - 9 different deviation values from 0.0 to 0.5
    - 20 time steps
    - Full smart loading hierarchy for efficient data management
    - Comprehensive analysis and visualization
    """
    
    print("🔬 Starting PRODUCTION static noise staggered quantum walk experiment...")
    print("=" * 80)
    
    # Production experiment parameters
    N = 200
    theta = np.pi / 4
    steps = 20
    samples = 50
    init_nodes = [N//2]
    deviation_ranges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    
    print(f"🎯 Production experiment parameters:")
    print(f"   N = {N} (quantum walk nodes)")
    print(f"   theta = π/4 (coin angle)")
    print(f"   steps = {steps} (time evolution)")
    print(f"   samples = {samples} (per deviation)")
    print(f"   init_nodes = {init_nodes} (starting position)")
    print(f"   deviation_ranges = {deviation_ranges}")
    print(f"   Total quantum walks to compute: {len(deviation_ranges)} × {samples} = {len(deviation_ranges) * samples}")
    print()
    
    # Create tessellation function for directory naming
    def static_noise_tesselation(N):
        """Tessellation identifier for static noise"""
        return None
    
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    start_time = time.time()
    
    # Use smart loading hierarchy for maximum efficiency
    print("🧠 Using smart load system for efficient data management...")
    print("   Level 1: Check for existing mean probability distributions")
    print("   Level 2: Check for existing sample data")
    print("   Level 3: Run new experiments as needed")
    print()
    
    # Step 1: Try to load mean probability distributions
    print("📊 Step 1: Checking for existing mean probability distributions...")
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
        
        experiment_time = time.time() - start_time
        print(f"⚡ Loaded existing results in {experiment_time:.2f} seconds!")
        
    else:
        # Step 2: Check for samples and create probability distributions  
        print("📂 Step 2: Checking for existing sample data...")
        samples_exist = check_all_samples_exist(N, theta, deviation_ranges, samples, steps)
        
        if samples_exist:
            print("✅ Found existing samples - creating mean probability distributions...")
            probdist_start = time.time()
            create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples)
            mean_results = load_mean_probability_distributions(
                static_noise_tesselation, N, steps, deviation_ranges,
                "experiments_data_samples_probDist", "static_noise"
            )
            final_results = [result[-1] if result else None for result in mean_results]
            
            probdist_time = time.time() - probdist_start
            experiment_time = time.time() - start_time
            print(f"✅ Created probability distributions in {probdist_time:.2f} seconds!")
            
        else:
            # Step 3: Run new experiment
            print("🔄 Step 3: No existing data found - running NEW experiment...")
            print(f"   This will compute {len(deviation_ranges) * samples} quantum walks")
            print(f"   Estimated time: ~{len(deviation_ranges) * samples * 0.1:.1f} seconds")
            print()
            
            experiment_start = time.time()
            final_results = run_static_noise_samples_experiment(
                N, theta, steps, init_nodes, deviation_ranges, samples
            )
            
            experiment_time = time.time() - start_time
            actual_compute_time = time.time() - experiment_start
            print(f"⚡ Computation completed in {actual_compute_time:.2f} seconds!")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Static noise experiment COMPLETED in {total_time:.2f} seconds")
    print(f"   Processed {len(deviation_ranges)} deviations × {samples} samples = {len(deviation_ranges) * samples} quantum walks")
    print()
    
    # Analysis and visualization
    print("📈 Performing comprehensive analysis...")
    try:
        analysis_start = time.time()
        analysis_results = analyze_static_noise_results(final_results, deviation_ranges, N, steps)
        
        print("🎨 Creating publication-quality plots...")
        create_static_noise_plots(analysis_results, deviation_ranges, N, theta, steps, samples)
        
        print("💾 Saving detailed results...")
        save_static_noise_results(analysis_results, deviation_ranges, N, theta, steps, samples)
        
        analysis_time = time.time() - analysis_start
        print(f"✅ Analysis and visualization completed in {analysis_time:.2f} seconds!")
        
    except Exception as e:
        print(f"⚠️  Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("🏆 PRODUCTION EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"📊 Processed {len(deviation_ranges)} noise deviations: {deviation_ranges}")
    print(f"🔬 Total quantum walks computed: {len(deviation_ranges) * samples}")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds")
    print(f"📈 Average time per quantum walk: {total_time / (len(deviation_ranges) * samples):.4f} seconds")
    print(f"💾 Data saved in experiments_data_samples/ and experiments_data_samples_probDist/")
    print(f"📊 Analysis results saved in static_noise_results/")
    print("=" * 80)
    
    return {
        "deviation_ranges": deviation_ranges,
        "N": N, "theta": theta, "steps": steps, "samples": samples,
        "total_time": total_time,
        "final_results": final_results,
        "total_quantum_walks": len(deviation_ranges) * samples,
        "avg_time_per_walk": total_time / (len(deviation_ranges) * samples)
    }


def check_all_samples_exist(N, theta, deviation_ranges, samples, steps):
    """Check if all sample files exist for all deviations"""
    
    def static_noise_tesselation(N):
        return None
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    print(f"   Checking {len(deviation_ranges)} deviations × {samples} samples × {steps} steps...")
    
    missing_count = 0
    total_files = len(deviation_ranges) * samples * steps
    
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
                    missing_count += 1
    
    existing_files = total_files - missing_count
    print(f"   Found {existing_files}/{total_files} existing files ({existing_files/total_files*100:.1f}%)")
    
    return missing_count == 0


def run_static_noise_samples_experiment(N, theta, steps, init_nodes, deviation_ranges, samples):
    """Run the static noise experiment with immediate sample saving and progress tracking"""
    
    print(f"🚀 RUNNING static noise experiment...")
    print(f"   {samples} samples for each of {len(deviation_ranges)} deviations")
    print(f"   Total quantum walks: {len(deviation_ranges) * samples}")
    print()
    
    def static_noise_tesselation(N):
        return None
    static_noise_tesselation.__name__ = "static_noise_tesselation"
    
    final_results = []
    total_walks_computed = 0
    total_walks_loaded = 0
    
    for dev_idx, deviation in enumerate(tqdm(deviation_ranges, desc="Deviations", position=0)):
        print(f"\n{'='*60}")
        print(f"🎯 Processing deviation {deviation:.3f} ({dev_idx+1}/{len(deviation_ranges)})")
        print(f"{'='*60}")
        
        # Create experiment directory following standard pattern
        has_noise = deviation > 0
        noise_params = [deviation]
        exp_dir = get_experiment_dir(
            static_noise_tesselation, has_noise, N, noise_params, "static_noise", "experiments_data_samples"
        )
        
        dev_samples = []
        dev_computed = 0
        dev_loaded = 0
        
        for sample_idx in tqdm(range(samples), desc=f"Samples (dev={deviation:.3f})", position=1, leave=False):
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
                dev_loaded += 1
                total_walks_loaded += 1
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
                dev_computed += 1
                total_walks_computed += 1
        
        # Calculate mean for this deviation
        mean_prob = np.mean(dev_samples, axis=0)
        final_results.append(mean_prob)
        
        print(f"✅ Deviation {deviation:.3f} completed:")
        print(f"   📊 {len(dev_samples)} total samples processed")
        print(f"   🔄 {dev_computed} new quantum walks computed")
        print(f"   📂 {dev_loaded} samples loaded from cache")
    
    print(f"\n🎉 ALL SAMPLES COMPLETED!")
    print(f"   🔄 Total new quantum walks computed: {total_walks_computed}")
    print(f"   📂 Total samples loaded from cache: {total_walks_loaded}")
    print(f"   📊 Total samples processed: {total_walks_computed + total_walks_loaded}")
    
    # Create mean probability distributions using the standard system
    print("\n📊 Creating mean probability distributions in standard format...")
    probdist_start = time.time()
    create_static_noise_probdist_from_samples(N, theta, steps, deviation_ranges, samples)
    probdist_time = time.time() - probdist_start
    print(f"✅ Mean probability distributions created in {probdist_time:.2f} seconds")
    
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
    """Comprehensive analysis of static noise experiment results"""
    
    print("🔬 Performing detailed static noise analysis...")
    
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
    
    print(f"   📊 Analyzing {len(deviation_ranges)} different noise levels...")
    
    for i, (dev, mean_prob) in enumerate(zip(deviation_ranges, mean_results)):
        # Maximum probability and location
        max_prob = np.max(mean_prob)
        max_node = np.argmax(mean_prob)
        analysis['max_probabilities'].append(max_prob)
        analysis['max_prob_nodes'].append(max_node)
        
        # Total variation from uniform distribution
        total_var = np.sum(np.abs(mean_prob - uniform_prob))
        analysis['total_variations'].append(total_var)
        
        # Position spread (standard deviation)
        prob_normalized = mean_prob / np.sum(mean_prob)
        mean_position = np.sum(domain * prob_normalized.flatten())
        variance = np.sum(((domain - mean_position) ** 2) * prob_normalized.flatten())
        spread = np.sqrt(variance)
        analysis['spreads'].append(spread)
        
        # Shannon entropy
        prob_clean = mean_prob[mean_prob > 1e-12]
        entropy = -np.sum(prob_clean * np.log2(prob_clean))
        analysis['entropy'].append(entropy)
        
        print(f"   🎯 Deviation {dev:.3f}: max_prob={max_prob:.4f} at node {max_node}, "
              f"total_var={total_var:.4f}, spread={spread:.3f}, entropy={entropy:.3f}")
    
    print("✅ Analysis completed!")
    return analysis


def create_static_noise_plots(analysis_results, deviation_ranges, N, theta, steps, samples):
    """Create publication-quality plots for static noise analysis"""
    
    print("🎨 Creating comprehensive analysis plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Mean probabilities vs position for different noise levels
    domain = np.arange(N) - N//2
    colors = plt.cm.viridis(np.linspace(0, 1, len(deviation_ranges)))
    
    for i, (dev, mean_prob) in enumerate(zip(deviation_ranges, analysis_results['mean_probabilities'])):
        ax1.plot(domain, mean_prob.flatten(), color=colors[i], 
                label=f'σ={dev:.3f}', alpha=0.8, linewidth=2)
    
    ax1.set_xlabel('Position', fontsize=12)
    ax1.set_ylabel('Mean Probability', fontsize=12)
    ax1.set_title(f'Probability Distributions vs Position\n(N={N}, θ=π/4, steps={steps}, samples={samples})', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Maximum probability vs noise deviation
    ax2.plot(deviation_ranges, analysis_results['max_probabilities'], 'ro-', 
             linewidth=3, markersize=8, markerfacecolor='red', markeredgecolor='darkred')
    ax2.set_xlabel('Noise Deviation (σ)', fontsize=12)
    ax2.set_ylabel('Maximum Probability', fontsize=12)
    ax2.set_title('Peak Probability vs Noise Level', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Total variation vs noise deviation
    ax3.plot(deviation_ranges, analysis_results['total_variations'], 'bo-', 
             linewidth=3, markersize=8, markerfacecolor='blue', markeredgecolor='darkblue')
    ax3.set_xlabel('Noise Deviation (σ)', fontsize=12)
    ax3.set_ylabel('Total Variation from Uniform', fontsize=12)
    ax3.set_title('Departure from Uniformity vs Noise', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Position spread vs noise deviation
    ax4.plot(deviation_ranges, analysis_results['spreads'], 'go-', 
             linewidth=3, markersize=8, markerfacecolor='green', markeredgecolor='darkgreen')
    ax4.set_xlabel('Noise Deviation (σ)', fontsize=12)
    ax4.set_ylabel('Position Spread (σ_pos)', fontsize=12)
    ax4.set_title('Spatial Spreading vs Noise Level', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # Save with high quality
    plot_filename = f"static_noise_production_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Analysis plots saved to: {plot_filename}")
    
    plt.show()


def save_static_noise_results(analysis_results, deviation_ranges, N, theta, steps, samples):
    """Save comprehensive results with detailed metadata"""
    
    results_dir = "static_noise_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save complete analysis as pickle with metadata
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_data = {
        'analysis_results': analysis_results,
        'parameters': {
            'N': N, 'theta': theta, 'steps': steps, 'samples': samples,
            'deviation_ranges': deviation_ranges
        },
        'metadata': {
            'timestamp': timestamp,
            'total_quantum_walks': len(deviation_ranges) * samples,
            'experiment_type': 'static_noise_production'
        }
    }
    
    analysis_filename = os.path.join(results_dir, f"static_noise_production_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}_{timestamp}.pkl")
    with open(analysis_filename, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    # Save summary as CSV with enhanced information
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
        
        # Add metadata as header comment
        csv_filename = os.path.join(results_dir, f"static_noise_production_N{N}_theta{theta:.4f}_steps{steps}_samples{samples}_{timestamp}.csv")
        with open(csv_filename, 'w') as f:
            f.write(f"# Static Noise Production Experiment Results\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"# Parameters: N={N}, theta={theta:.4f}, steps={steps}, samples={samples}\n")
            f.write(f"# Total quantum walks: {len(deviation_ranges) * samples}\n")
            f.write(f"# Deviation range: {min(deviation_ranges):.3f} to {max(deviation_ranges):.3f}\n")
            f.write(f"#\n")
        
        df.to_csv(csv_filename, mode='a', index=False)
        
        print(f"✅ Results saved to:")
        print(f"   📊 Analysis: {analysis_filename}")
        print(f"   📈 Summary: {csv_filename}")
        
    except ImportError:
        print(f"✅ Results saved to:")
        print(f"   📊 Analysis: {analysis_filename}")
        print("   📈 CSV summary skipped (pandas not available)")


if __name__ == "__main__":
    print("🚀 STATIC NOISE PRODUCTION EXPERIMENT")
    print("=" * 80)
    print("This is the production version with full parameters:")
    print("- N=200 nodes, 50 samples per deviation")
    print("- 9 deviation levels from 0.0 to 0.5")
    print("- Smart loading for maximum efficiency")
    print("- Comprehensive analysis and visualization")
    print("=" * 80)
    print()
    
    try:
        result = run_static_noise_production_experiment()
        
        print("\n🎉 PRODUCTION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"📊 Processed {result['total_quantum_walks']} quantum walks")
        print(f"⏱️  Total time: {result['total_time']:.2f} seconds")
        print(f"📈 Average: {result['avg_time_per_walk']:.4f} seconds per quantum walk")
        
    except Exception as e:
        print(f"\n❌ Production experiment FAILED: {e}")
        import traceback
        traceback.print_exc()
