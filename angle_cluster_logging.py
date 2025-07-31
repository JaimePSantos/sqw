#!/usr/bin/env python3
"""
Angle noise experiment with both cluster deployment and crash-safe logging.
Combines cluster_module decorator for deployment automation with logging_module for crash protection.
"""

import time
import numpy as np
from logging_module import crash_safe_log


@crash_safe_log(
    log_file_prefix="angle_cluster_experiment",
    heartbeat_interval=30.0,
    log_system_info=True
)
def run_angle_experiment_with_logging():
    """
    Run angle noise quantum walk experiment with cluster deployment and crash-safe logging.
    
    Features:
    - Cluster deployment automation (virtual env, dependency checking, result bundling)
    - Crash-safe logging (separate process, signal handling, organized logs)
    - Immediate sample saving for interruption recovery
    - Progress tracking with ETA calculations
    """
    
    print("🔥 DEBUG: Function called!")
    
    # Import after cluster environment is set up
    import logging
    logger = logging.getLogger(__name__)
    
    print("🔥 DEBUG: Logger created!")
    logger.info("🔥 FUNCTION STARTED - Testing logging")
    logger.info("🔥 About to try imports")
    
    print("🔥 DEBUG: About to try imports!")
    
    try:
        import numpy as np
        import networkx as nx
        import pickle
        import os
        
        print("🔥 DEBUG: Basic imports successful!")
        
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        
        print("🔥 DEBUG: SQW imports successful!")
        
        # Import shared functions from jaime_scripts
        from jaime_scripts import (
            get_experiment_dir,
            create_mean_probability_distributions,
            load_mean_probability_distributions,
            prob_distributions2std
        )
        
        print("🔥 DEBUG: jaime_scripts imports successful!")
        
        logger.info("Successfully imported all required modules")
        
    except ImportError as e:
        print(f"🔥 DEBUG: Import error: {e}")
        logger.error(f"Could not import required modules: {e}")
        logger.error("Make sure you're running this script from the correct directory with all dependencies available")
        raise
    except Exception as e:
        print(f"🔥 DEBUG: Unexpected error: {e}")
        logger.error(f"Unexpected error during imports: {e}")
        raise

    logger.info("Starting angle noise quantum walk experiment with cluster deployment and crash-safe logging")
    
    # Cluster-optimized parameters (matching existing data)
    N = 2000  # System size 
    steps = 500  # Time steps (up to step_499 exists)
    samples = 10  # Samples per angle deviation (samples 0-9 exist)
    initial_state_kwargs = {"nodes": [N//2]}
    tesselation_order = [[0,1] for x in range(steps)]  # Fixed tesselation order for angle experiments

    # List of angle noise deviations (starting simple)
    angle_devs = [0]  # Just no-noise case first
    
    logger.info(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    logger.info(f"Angle deviations: {[f'{dev:.3f}' for dev in angle_devs]}")
    logger.info("🚀 IMMEDIATE SAVE MODE: Each sample will be saved as soon as it's computed!")
    logger.info("🛡️ CRASH-SAFE LOGGING: All progress logged to organized date-based files")
    
    print(f"🔥 DEBUG: Parameters set - N={N}, steps={steps}, samples={samples}")
    print(f"🔥 DEBUG: Angle devs = {angle_devs}")
    
    logger.error("🔥 DEBUG: FORCED ERROR LOG - Parameters set!")
    logger.warning("🔥 DEBUG: FORCED WARNING LOG - About to start timing!")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(angle_devs) * samples
    completed_samples = 0

    print("🔥 DEBUG: About to start main experiment loop!")
    logger.info("🔥 DEBUG: About to start main experiment loop!")
    
    # Run experiments with immediate saving for each sample
    for dev_idx, angle_dev in enumerate(angle_devs):
        print(f"🔥 DEBUG: Starting angle deviation {angle_dev}")
        logger.info(f"🔥 DEBUG: Starting angle deviation {angle_dev}")
        logger.info(f"=== Processing angle deviation {angle_dev:.3f} ({dev_idx+1}/{len(angle_devs)}) ===")
        
        # Setup experiment directory - use the correct existing structure
        has_noise = angle_dev > 0
        if has_noise:
            exp_dir = os.path.join("experiments_data_samples", "even_line_two_tesselation_angle_noise", f"N_{N}")
        else:
            exp_dir = os.path.join("experiments_data_samples", "even_line_two_tesselation_angle_nonoise", f"N_{N}")
        os.makedirs(exp_dir, exist_ok=True)
        logger.info(f"Experiment directory: {exp_dir}")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        
        for sample_idx in range(samples):
            sample_start_time = time.time()
            
            # Debug: Log the path we're checking
            logger.info(f"Checking if sample {sample_idx+1}/{samples} exists in {exp_dir}")
            
            # Check if this sample already exists (all step files)
            sample_exists = True
            missing_steps = []
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    missing_steps.append(step_idx)
                    if len(missing_steps) <= 5:  # Only log first few missing steps
                        logger.debug(f"Missing: {filepath}")
            
            if sample_exists:
                logger.info(f"✅ Sample {sample_idx+1}/{samples} already exists (all {steps} steps found), skipping")
                completed_samples += 1
                continue
            else:
                logger.info(f"❌ Sample {sample_idx+1}/{samples} missing {len(missing_steps)} steps, needs computation")
            
            logger.info(f"Computing sample {sample_idx+1}/{samples} for angle deviation {angle_dev:.3f}")
            
            # Generate angles for this sample
            if angle_dev == 0:
                # No noise case
                angles = [[np.pi/3, np.pi/3]] * steps
            else:
                # Add noise to angles
                angles = []
                for _ in range(steps):
                    noise1 = np.random.uniform(-angle_dev, angle_dev)
                    noise2 = np.random.uniform(-angle_dev, angle_dev)
                    angles.append([np.pi/3 + noise1, np.pi/3 + noise2])
            
            # Run the quantum walk experiment for this sample
            graph = nx.cycle_graph(N)
            tesselation = even_line_two_tesselation(N)
            initial_state = uniform_initial_state(N, **initial_state_kwargs)
            
            logger.debug(f"Running quantum walk with {steps} steps")
            print(f"🔥 DEBUG: About to call running() for sample {sample_idx}")
            
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
            
            dev_computed_samples += 1
            completed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Progress report
            progress_pct = (completed_samples / total_samples) * 100
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_samples / completed_samples if completed_samples > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > elapsed_time else 0
            
            logger.info(f"✅ Sample {sample_idx+1}/{samples} saved in {sample_time:.1f}s")
            logger.info(f"Progress: {completed_samples}/{total_samples} ({progress_pct:.1f}%), Elapsed: {elapsed_time:.1f}s, Remaining: ~{remaining_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        logger.info(f"✅ Angle deviation {angle_dev:.3f} completed: {dev_computed_samples} new samples in {dev_time:.1f}s")

    experiment_time = time.time() - start_time
    logger.info(f"🎉 All samples completed in {experiment_time:.2f} seconds")
    logger.info(f"Total samples computed: {completed_samples}")
    
    # Create mean probability distributions
    logger.info("📊 Creating mean probability distributions from saved samples...")
    try:
        create_mean_probability_distributions(
            even_line_two_tesselation, N, steps, angle_devs, samples, 
            "experiments_data_samples", "experiments_data_samples_probDist", "angle"
        )
        logger.info("✅ Mean probability distributions created successfully!")
    except Exception as e:
        logger.warning(f"Could not create mean probability distributions: {e}")
        logger.warning("You can create them later using the saved sample data.")
    
    # Load and validate results
    try:
        mean_results = load_mean_probability_distributions(
            even_line_two_tesselation, N, steps, angle_devs, "experiments_data_samples_probDist", "angle"
        )
        logger.info("✅ Mean probability distributions loaded for analysis")
        
        # Calculate statistics for verification
        domain = np.arange(N) - N//2  # Centered domain for angle experiments
        stds = []
        for i, dev_mean_prob_dists in enumerate(mean_results):
            if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
                std_values = prob_distributions2std(dev_mean_prob_dists, domain)
                stds.append(std_values)
                logger.info(f"Angle dev {angle_devs[i]:.3f}: Final std = {std_values[-1]:.3f}")
            else:
                stds.append([])
                logger.warning(f"Angle dev {angle_devs[i]:.3f}: No valid probability distributions found")
        
    except Exception as e:
        logger.warning(f"Could not load mean probability distributions: {e}")

    logger.info("Angle noise experiment completed successfully!")
    total_time = time.time() - start_time
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    
    # Log performance summary
    logger.info("=== Performance Summary ===")
    logger.info(f"System size (N): {N}")
    logger.info(f"Time steps: {steps}")
    logger.info(f"Samples per angle deviation: {samples}")
    logger.info(f"Number of angle deviations: {len(angle_devs)}")
    logger.info(f"Total quantum walks computed: {len(angle_devs) * samples}")
    if experiment_time > 0:
        logger.info(f"Average time per quantum walk: {experiment_time / (len(angle_devs) * samples):.3f} seconds")
    
    logger.info("=== Angle Noise Details ===")
    logger.info("Angle noise model:")
    logger.info("- angle_dev=0: Fixed angles [π/3, π/3] (no noise)")
    logger.info("- angle_dev>0: Random noise added to each angle uniformly in [-dev, +dev]")
    logger.info("- Each sample generates different random angle sequences")
    logger.info("- Mean probability distributions average over all samples")
    
    return {
        "angle_devs": angle_devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time,
        "stds": stds if 'stds' in locals() else []
    }


if __name__ == "__main__":
    run_angle_experiment_with_logging()
