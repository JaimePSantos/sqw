#!/usr/bin/env python3
"""
Clean cluster-compatible angle noise experiment using the cluster decorator module.
This version eliminates all duplicate cluster management code by using the cluster_deploy decorator.
"""

import time
import numpy as np
from cluster_module import cluster_deploy


@cluster_deploy(
    experiment_name="angle_noise",
    noise_type="angle",
    N=2000,
    samples=10
)
def run_angle_experiment():
    """Run the angle noise quantum walk experiment with immediate saving."""
    
    # Import after cluster environment is set up
    try:
        import numpy as np
        import networkx as nx
        import pickle
        import os
        
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.utils import random_angle_deviation
        
        # Import shared functions from jaime_scripts
        from jaime_scripts import (
            get_experiment_dir,
            create_mean_probability_distributions,
            load_mean_probability_distributions,
            prob_distributions2std
        )
        
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        raise

    print("Starting angle noise quantum walk experiment...")
    
    # Cluster-optimized parameters
    N = 2000  # System size
    steps = N//4  # Time steps
    samples = 10  # Samples per deviation
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # List of angle noise deviations
    devs = [0, (np.pi/3)/2.5, (np.pi/3)*2]
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print("🚀 IMMEDIATE SAVE MODE: Each sample will be saved as soon as it's computed!")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0

    # Run experiments with immediate saving for each sample
    for dev_idx, dev in enumerate(devs):
        print(f"\n=== Processing angle deviation {dev:.4f} ({dev_idx+1}/{len(devs)}) ===")
        
        # Setup experiment directory
        has_noise = dev > 0
        noise_params = [dev, dev] if has_noise else [0, 0]
        exp_dir = get_experiment_dir(even_line_two_tesselation, has_noise, N, noise_params=noise_params, noise_type="angle", base_dir="experiments_data_samples")
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
            
            # Generate angle sequence for this sample
            if dev == 0:
                # No noise case - use perfect angles
                sample_angles = [[np.pi/3, np.pi/3]] * steps
            else:
                sample_angles = random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps)
            
            # Run the quantum walk experiment for this sample
            graph = nx.cycle_graph(N)
            tesselation = even_line_two_tesselation(N)
            initial_state = uniform_initial_state(N, **initial_state_kwargs)
            
            # Run the walk
            walk_result = running(
                graph, tesselation, steps,
                initial_state,
                angles=sample_angles,
                tesselation_order=tesselation_order
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
        print(f"✅ Angle deviation {dev:.4f} completed: {dev_computed_samples} new samples in {dev_time:.1f}s")

    experiment_time = time.time() - start_time
    print(f"\n🎉 All samples completed in {experiment_time:.2f} seconds")
    print(f"Total samples computed: {completed_samples}")
    
    # Create mean probability distributions
    print("\n📊 Creating mean probability distributions from saved samples...")
    try:
        create_mean_probability_distributions(
            even_line_two_tesselation, N, steps, devs, samples, 
            "experiments_data_samples", "experiments_data_samples_probDist", "angle"
        )
        print("✅ Mean probability distributions created successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not create mean probability distributions: {e}")
        print("   You can create them later using the saved sample data.")
    
    # Load and validate results
    try:
        mean_results = load_mean_probability_distributions(
            even_line_two_tesselation, N, steps, devs, "experiments_data_samples_probDist", "angle"
        )
        print("✅ Mean probability distributions loaded for analysis")
        
        # Calculate statistics for verification
        domain = np.arange(N) - N//2  # Center domain around 0
        stds = []
        for i, dev_mean_prob_dists in enumerate(mean_results):
            if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
                std_values = prob_distributions2std(dev_mean_prob_dists, domain)
                stds.append(std_values)
                print(f"Dev {devs[i]:.3f}: Final std = {std_values[-1]:.3f}")
            else:
                stds.append([])
                print(f"Dev {devs[i]:.3f}: No valid probability distributions found")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load mean probability distributions: {e}")

    print("Angle experiment completed successfully!")
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
    
    print("\n=== Angle Noise Details ===")
    print("Angle noise model:")
    print("- dev=0: Perfect angles [π/3, π/3] (no noise)")
    print("- dev>0: Random deviation from [π/3, π/3] with standard deviation 'dev'")
    print("- Each sample generates a different random angle sequence")
    print("- Mean probability distributions average over all samples")
    
    return {
        "devs": devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time
    }


if __name__ == "__main__":
    run_angle_experiment()
