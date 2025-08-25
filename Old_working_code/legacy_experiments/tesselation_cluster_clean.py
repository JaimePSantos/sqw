#!/usr/bin/env python3
"""
Clean cluster-compatible tesselation order experiment using the cluster decorator module.
This version eliminates all duplicate cluster management code by using the cluster_deploy decorator.
"""

import time
import numpy as np
from cluster_module import cluster_deploy


@cluster_deploy(
    experiment_name="tesselation_order",
    noise_type="tesselation_order",
    N=2000,
    samples=10
)
def run_tesselation_experiment():
    """Run the tesselation order quantum walk experiment with immediate saving."""
    
    # Import after cluster environment is set up
    try:
        import numpy as np
        import networkx as nx
        import pickle
        import os
        
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.utils import tesselation_choice
        
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

    print("Starting tesselation order quantum walk experiment...")
    
    # Cluster-optimized parameters
    N = 2000  # System size
    steps = N//4  # Time steps
    samples = 10  # Samples per shift probability
    angles = [[np.pi/3, np.pi/3]] * steps  # Fixed angles, no noise
    initial_state_kwargs = {"nodes": [N//2]}

    # List of tesselation shift probabilities
    shift_probs = [0, 0.2, 0.5]
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print("🚀 IMMEDIATE SAVE MODE: Each sample will be saved as soon as it's computed!")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(shift_probs) * samples
    completed_samples = 0

    # Run experiments with immediate saving for each sample
    for shift_prob_idx, shift_prob in enumerate(shift_probs):
        print(f"\n=== Processing shift probability {shift_prob:.3f} ({shift_prob_idx+1}/{len(shift_probs)}) ===")
        
        # Setup experiment directory
        has_noise = shift_prob > 0
        noise_params = [shift_prob] if has_noise else [0]
        exp_dir = get_experiment_dir(even_line_two_tesselation, has_noise, N, noise_params=noise_params, noise_type="tesselation_order", base_dir="experiments_data_samples")
        os.makedirs(exp_dir, exist_ok=True)
        
        shift_start_time = time.time()
        shift_computed_samples = 0
        
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
            
            # Generate tesselation order for this sample
            if shift_prob == 0:
                # No noise case - use fixed tesselation order
                tesselation_order = [[0, 1]] * steps
            else:
                # Noisy tesselation order - generate different random sequence for each sample
                tesselation_order = tesselation_choice([[0, 1], [1, 0]], steps, [1 - shift_prob, shift_prob])
            
            # Run the quantum walk experiment for this sample
            graph = nx.cycle_graph(N)
            tesselation = even_line_two_tesselation(N)
            initial_state = uniform_initial_state(N, **initial_state_kwargs)
            
            # Run the walk
            walk_result = running(
                graph, tesselation, steps,
                initial_state,
                angles=angles,
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
            
            shift_computed_samples += 1
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
        
        shift_time = time.time() - shift_start_time
        print(f"✅ Shift probability {shift_prob:.3f} completed: {shift_computed_samples} new samples in {shift_time:.1f}s")

    experiment_time = time.time() - start_time
    print(f"\n🎉 All samples completed in {experiment_time:.2f} seconds")
    print(f"Total samples computed: {completed_samples}")
    
    # Create mean probability distributions
    print("\n📊 Creating mean probability distributions from saved samples...")
    try:
        create_mean_probability_distributions(
            even_line_two_tesselation, N, steps, shift_probs, samples, 
            "experiments_data_samples", "experiments_data_samples_probDist", "tesselation_order"
        )
        print("✅ Mean probability distributions created successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not create mean probability distributions: {e}")
        print("   You can create them later using the saved sample data.")
    
    # Load and validate results
    try:
        mean_results = load_mean_probability_distributions(
            even_line_two_tesselation, N, steps, shift_probs, "experiments_data_samples_probDist", "tesselation_order"
        )
        print("✅ Mean probability distributions loaded for analysis")
        
        # Calculate statistics for verification
        domain = np.arange(N)  # Use full domain for tesselation experiments
        stds = []
        for i, shift_mean_prob_dists in enumerate(mean_results):
            if shift_mean_prob_dists and len(shift_mean_prob_dists) > 0 and all(state is not None for state in shift_mean_prob_dists):
                std_values = prob_distributions2std(shift_mean_prob_dists, domain)
                stds.append(std_values)
                print(f"Shift prob {shift_probs[i]:.3f}: Final std = {std_values[-1]:.3f}")
            else:
                stds.append([])
                print(f"Shift prob {shift_probs[i]:.3f}: No valid probability distributions found")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not load mean probability distributions: {e}")

    print("Tesselation experiment completed successfully!")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n=== Performance Summary ===")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per shift probability: {samples}")
    print(f"Number of shift probabilities: {len(shift_probs)}")
    print(f"Total quantum walks computed: {len(shift_probs) * samples}")
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(shift_probs) * samples):.3f} seconds")
    
    print("\n=== Tesselation Order Details ===")
    print("Tesselation order noise model:")
    print("- shift_prob=0: Fixed order [[0,1], [0,1], ...] (no noise)")
    print("- shift_prob>0: Random switching between [[0,1], [1,0]] with given probability")
    print("- Each sample generates a different random tesselation sequence")
    print("- Mean probability distributions average over all samples")
    
    return {
        "shift_probs": shift_probs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time
    }


if __name__ == "__main__":
    run_tesselation_experiment()
