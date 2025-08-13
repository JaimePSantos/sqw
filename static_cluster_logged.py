#!/usr/bin/env python3

"""
Clean cluster-compatible static noise experiment using the cluster decorator module.

This version eliminates all duplicate cluster management code by using the cluster_deploy decorator.

Now uses smart loading from smart_loading_static module.
"""

import time
import math

# Import crash-safe logging decorator
from logging_module.crash_safe_logging import crash_safe_log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Cluster module switch
USE_CLUSTER = True  # Set to False to run locally without cluster deployment

# Plotting switch
ENABLE_PLOTTING = False  # Set to False to disable plotting
USE_LOGLOG_PLOT = False  # Set to True to use log-log scale for plotting

# Experiment parameters
N = 300  # System size
steps = N//4  # Time steps
samples = 10  # Samples per deviation

# Quantum walk parameters (for static noise, we only need theta)
theta = math.pi/3  # Base theta parameter for static noise
initial_state_kwargs = {"nodes": [N//2]}

# List of static noise deviations
devs = [0, 0.1, 0.5,10,100]

# Note: Set USE_LOGLOG_PLOT = True in the plotting configuration above to use log-log scale
# This is useful for identifying power-law behavior in the standard deviation growth

# ============================================================================
# CLUSTER DEPLOYMENT SETUP
# ============================================================================

if USE_CLUSTER:
    from cluster_module import cluster_deploy
else:
    # Mock decorator for local execution
    def cluster_deploy(**kwargs):
        def decorator(func):
            return func
        return decorator

# ============================================================================
# STANDARD DEVIATION DATA MANAGEMENT
# ============================================================================

def create_or_load_std_data(mean_results, devs, N, steps, tesselation_func, std_base_dir, noise_type):
    """
    Create or load standard deviation data from mean probability distributions.
    
    Args:
        mean_results: List of mean probability distributions for each parameter
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        tesselation_func: Function to create tesselation (dummy for static noise)
        std_base_dir: Base directory for standard deviation data
        noise_type: Type of noise ("static_noise")
    
    Returns:
        List of standard deviation arrays for each deviation value
    """
    import os
    import pickle
    import numpy as np
    
    # Import functions from jaime_scripts and smart_loading_static
    from jaime_scripts import (
        prob_distributions2std
    )
    from smart_loading_static import get_experiment_dir
    
    print(f"\n📊 Managing standard deviation data in '{std_base_dir}'...")
    
    # Create base directory for std data
    os.makedirs(std_base_dir, exist_ok=True)
    
    stds = []
    domain = np.arange(N) - N//2  # Center domain around 0
    
    for i, dev in enumerate(devs):
        # Setup std data directory structure for static noise
        has_noise = dev > 0
        noise_params = [dev] if has_noise else [0]  # Static noise uses single parameter
        std_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type=noise_type, 
                                   base_dir=std_base_dir)
        os.makedirs(std_dir, exist_ok=True)
        
        std_filepath = os.path.join(std_dir, "std_vs_time.pkl")
        
        # Try to load existing std data
        if os.path.exists(std_filepath):
            try:
                with open(std_filepath, 'rb') as f:
                    std_values = pickle.load(f)
                print(f"  ✅ Loaded std data for dev {dev:.3f}")
                stds.append(std_values)
                continue
            except Exception as e:
                print(f"  ⚠️  Could not load std data for dev {dev:.3f}: {e}")
        
        # Compute std data from mean probability distributions
        print(f"  🔄 Computing std data for dev {dev:.3f}...")
        try:
            # Get mean probability distributions for this deviation
            if mean_results and i < len(mean_results) and mean_results[i]:
                dev_mean_prob_dists = mean_results[i]
                
                # Calculate standard deviations
                if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
                    std_values = prob_distributions2std(dev_mean_prob_dists, domain)
                    
                    # Save std data
                    with open(std_filepath, 'wb') as f:
                        pickle.dump(std_values, f)
                    
                    stds.append(std_values)
                    print(f"  ✅ Computed and saved std data for dev {dev:.3f} (final std = {std_values[-1]:.3f})")
                else:
                    print(f"  ❌ No valid probability distributions found for dev {dev:.3f}")
                    stds.append([])
            else:
                print(f"  ❌ No mean results available for dev {dev:.3f}")
                stds.append([])
                
        except Exception as e:
            print(f"  ❌ Error computing std data for dev {dev:.3f}: {e}")
            stds.append([])
    
    print(f"✅ Standard deviation data management completed!")
    return stds

@crash_safe_log(log_file_prefix="static_noise_experiment", heartbeat_interval=30.0)
@cluster_deploy(
    experiment_name="static_noise",
    noise_type="static_noise",
    N=N,
    samples=samples
)
def run_static_experiment():
    """Run the static noise quantum walk experiment with immediate saving."""
    
    # Import after cluster environment is set up
    try:
        import numpy as np
        import networkx as nx
        import pickle
        import os
        
        # For static noise, we don't need tesselations module since it's built-in
        from sqw.experiments_expanded_static import running
        from sqw.states import uniform_initial_state, amp2prob
        
        # Import shared functions from jaime_scripts and smart_loading_static
        from jaime_scripts import (
            prob_distributions2std,
            plot_std_vs_time_qwak
        )
        from smart_loading_static import smart_load_or_create_experiment, get_experiment_dir
        
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        raise

    print("Starting static noise quantum walk experiment...")
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print("🚀 IMMEDIATE SAVE MODE: Each sample will be saved as soon as it's computed!")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0

    # Create a dummy tessellation function for static noise
    def dummy_tesselation_func(N):
        """Dummy tessellation function for static noise (tessellations are built-in)"""
        return None

    # Run experiments with immediate saving for each sample
    for dev_idx, dev in enumerate(devs):
        print(f"\n=== Processing static noise deviation {dev:.4f} ({dev_idx+1}/{len(devs)}) ===")
        
        # Setup experiment directory
        has_noise = dev > 0
        noise_params = [dev] if has_noise else [0]  # Static noise uses single parameter
        exp_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, noise_params=noise_params, noise_type="static_noise", base_dir="experiments_data_samples")
        os.makedirs(exp_dir, exist_ok=True)
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        
        for sample_idx in range(samples):
            sample_start_time = time.time()
            
            # Check if this sample already exists (all step files)
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                print(f"  ✅ Sample {sample_idx+1}/{samples} already exists, skipping")
                completed_samples += 1
                continue
            
            print(f"  🔄 Computing sample {sample_idx+1}/{samples}...")
            
            # For static noise, we don't need to generate angle sequences
            # The noise is applied internally by the running function
            deviation_range = dev
            
            # Extract initial nodes from initial_state_kwargs
            initial_nodes = initial_state_kwargs.get('nodes', [])
            
            # Run the quantum walk experiment for this sample using static noise
            walk_result = running(
                N, theta, steps,
                initial_nodes=initial_nodes,
                deviation_range=deviation_range,
                return_all_states=True
            )
            
            # Save each step immediately
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                os.makedirs(step_dir, exist_ok=True)
                
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(walk_result[step_idx], f)
            
            dev_computed_samples += 1
            completed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Progress report
            progress_pct = (completed_samples / total_samples) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_samples / completed_samples if completed_samples > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > elapsed_time else 0
            
            print(f"  ✅ Sample {sample_idx+1}/{samples} saved in {sample_time:.1f}s")
            print(f"     Progress: {completed_samples}/{total_samples} ({progress_pct:.1f}%)")
            print(f"     Elapsed: {elapsed_time:.1f}s, Remaining: ~{remaining_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        print(f"✅ Static noise deviation {dev:.4f} completed: {dev_computed_samples} new samples in {dev_time:.1f}s")

    experiment_time = time.time() - start_time
    print(f"\n🎉 All samples completed in {experiment_time:.2f} seconds")
    print(f"Total samples computed: {completed_samples}")
    
    # Smart load or create mean probability distributions
    print("\n📊 Smart loading/creating mean probability distributions...")
    try:
        mean_results = smart_load_or_create_experiment(
            graph_func=lambda n: None,  # Not used in static noise
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            angles_or_angles_list=theta,  # Single theta value for static noise
            tesselation_order_or_list=None,  # Not used in static noise
            initial_state_func=uniform_initial_state,
            initial_state_kwargs=initial_state_kwargs,
            parameter_list=devs,
            samples=samples,
            noise_type="static_noise",
            parameter_name="static_dev",
            samples_base_dir="experiments_data_samples",
            probdist_base_dir="experiments_data_samples_probDist"
        )
        print("✅ Mean probability distributions ready for analysis")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not smart load/create mean probability distributions: {e}")
        mean_results = None

    # Create or load standard deviation data
    try:
        stds = create_or_load_std_data(
            mean_results, devs, N, steps, dummy_tesselation_func,
            "experiments_data_samples_std", "static_noise"
        )
        
        # Print final std values for verification
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values and len(std_values) > 0:
                print(f"Dev {dev:.3f}: Final std = {std_values[-1]:.3f}")
            else:
                print(f"Dev {dev:.3f}: No valid standard deviation data")
                
    except Exception as e:
        print(f"⚠️  Warning: Could not create/load standard deviation data: {e}")
        stds = []

    # Plot standard deviation vs time if enabled
    if ENABLE_PLOTTING:
        print("\n📈 Creating standard deviation vs time plot...")
        try:
            if 'stds' in locals() and len(stds) > 0 and any(len(std) > 0 for std in stds):
                import matplotlib.pyplot as plt
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                for i, (std_values, dev) in enumerate(zip(stds, devs)):
                    if len(std_values) > 0:
                        time_steps = list(range(len(std_values)))
                        
                        # Filter out zero values for log-log plot
                        if USE_LOGLOG_PLOT:
                            # Remove zero values which can't be plotted on log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                            if filtered_data:
                                filtered_times, filtered_stds = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=f'Static deviation = {dev:.3f}', 
                                         marker='o', markersize=3, linewidth=2)


                        else:
                            plt.plot(time_steps, std_values, 
                                   label=f'Static deviation = {dev:.3f}', 
                                   marker='o', markersize=3, linewidth=2)
                
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                
                if USE_LOGLOG_PLOT:
                    plt.title('Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3, which="both", ls="-")  # Grid for both major and minor ticks
                    plot_filename = "static_noise_std_vs_time_loglog.png"
                else:
                    plt.title('Standard Deviation vs Time for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plot_filename = "static_noise_std_vs_time.png"
                
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved as '{plot_filename}'")
                
                # Show the plot
                plt.show()
                plot_type = "log-log" if USE_LOGLOG_PLOT else "linear"
                print(f"✅ Standard deviation plot displayed! (Scale: {plot_type})")
            else:
                print("⚠️  Warning: No standard deviation data available for plotting")
        except Exception as e:
            print(f"⚠️  Warning: Could not create plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n📈 Plotting disabled (ENABLE_PLOTTING=False)")

    print("Static noise experiment completed successfully!")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n=== Performance Summary ===")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(devs)}")
    print(f"Total quantum walks computed: {len(devs) * samples}")
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(devs) * samples):.3f} seconds")
    
    print("\n=== Static Noise Details ===")
    print("Static noise model:")
    print(f"- dev=0: Perfect static evolution with theta={theta:.3f} (no noise)")
    print("- dev>0: Random deviation applied to Hamiltonian edges with range 'dev'")
    print("- Each sample generates different random noise for edge parameters")
    print("- Mean probability distributions average over all samples")
    print("- Tessellations are built-in (alpha and beta patterns)")
    
    print("\n=== Plotting Features ===")
    print(f"- Plotting enabled: {ENABLE_PLOTTING}")
    if ENABLE_PLOTTING:
        plot_type = "Log-log scale" if USE_LOGLOG_PLOT else "Linear scale"
        plot_filename = "static_noise_std_vs_time_loglog.png" if USE_LOGLOG_PLOT else "static_noise_std_vs_time.png"
        print(f"- Plot type: {plot_type}")
        print(f"- Plot saved as: {plot_filename}")
        if USE_LOGLOG_PLOT:
            print("- Log-log plots help identify power-law scaling behavior σ(t) ∝ t^α")
    
    return {
        "devs": devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time,
        "theta": theta
    }

if __name__ == "__main__":
    run_static_experiment()