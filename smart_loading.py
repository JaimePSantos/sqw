"""
Smart Loading Module for Quantum Walk Experiments

This module contains the intelligent loading system that follows a 3-tier hierarchy:
1. Try to load mean probability distributions (fastest)
2. If not available, try to load samples and create probabilities  
3. If samples not available, run new experiment

Contains all necessary functions and dependencies for smart experiment loading.
"""

import os
import pickle
import time
import numpy as np
from sqw.states import amp2prob
from sqw.experiments_expanded import running

def get_experiment_dir(
    tesselation_func,
    has_noise,
    N,
    noise_params=None,
    noise_type="angle",  # "angle" or "tesselation_order"
    base_dir="experiments_data"
):
    """
    Returns the directory path for the experiment based on tesselation, noise, and graph size.
    """
    tesselation_name = tesselation_func.__name__
    if noise_type == "angle":
        noise_str = "angle_noise" if has_noise else "angle_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        if has_noise and noise_params is not None:
            # Round each noise param to 2 decimal places for folder name
            noise_suffix = "_".join(f"{float(x):.2f}" for x in noise_params)
            dev_folder = f"dev_{noise_suffix}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    elif noise_type == "tesselation_order":
        noise_str = "tesselation_order_noise" if has_noise else "tesselation_order_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        if has_noise and noise_params is not None:
            # Round each noise param to 3 decimal places for folder name
            noise_suffix = "_".join(f"{float(x):.3f}" for x in noise_params)
            shift_folder = f"tesselation_shift_prob_{noise_suffix}"
            return os.path.join(base, shift_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

def run_and_save_experiment_generic(
    graph_func,
    tesselation_func,
    N,
    steps,
    parameter_list,  # List of varying parameters for each walk
    angles_or_angles_list,  # Either fixed angles or list of angles for each walk
    tesselation_order_or_list,  # Either fixed tesselation_order or list for each walk
    initial_state_func,
    initial_state_kwargs,
    noise_params_list,  # List of noise parameters for each walk
    noise_type="angle",  # "angle" or "tesselation_order"
    parameter_name="dev",  # Name of the parameter for logging
    base_dir="experiments_data"
):
    """
    Generic function to run and save experiments for different parameter values.
    """
    results = []
    for i, (param, noise_params) in enumerate(zip(parameter_list, noise_params_list)):
        has_noise = any(p > 0 for p in noise_params) if isinstance(noise_params, list) else noise_params > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[run_and_save_experiment] Saving results to {exp_dir} for {parameter_name}={param:.3f}")

        G = graph_func(N)
        T = tesselation_func(N)
        initial_state = initial_state_func(N, **initial_state_kwargs)

        # Get the appropriate angles and tesselation_order for this walk
        if isinstance(angles_or_angles_list[0], list) and len(angles_or_angles_list) == len(parameter_list):
            angles = angles_or_angles_list[i]
        else:
            angles = angles_or_angles_list

        if isinstance(tesselation_order_or_list[0], list) and len(tesselation_order_or_list) == len(parameter_list):
            tesselation_order = tesselation_order_or_list[i]
        else:
            tesselation_order = tesselation_order_or_list

        print("[run_and_save_experiment] Running walk...")
        final_states = running(
            G, T, steps,
            initial_state,
            angles=angles,
            tesselation_order=tesselation_order
        )
        for j, state in enumerate(final_states):
            filename = f"final_state_step_{j}.pkl"
            with open(os.path.join(exp_dir, filename), "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[run_and_save_experiment] Saved {len(final_states)} states for {parameter_name}={param:.3f}.")
        results.append(final_states)
    return results

def load_experiment_results_generic(
    tesselation_func,
    N,
    steps,
    parameter_list,
    noise_params_list,
    noise_type="angle",
    base_dir="experiments_data"
):
    """
    Generic function to load experiment results from disk.
    """
    all_results = []
    for param, noise_params in zip(parameter_list, noise_params_list):
        has_noise = any(p > 0 for p in noise_params) if isinstance(noise_params, list) else noise_params > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        walk_results = []
        for i in range(steps):
            filename = f"final_state_step_{i}.pkl"
            filepath = os.path.join(exp_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    walk_results.append(pickle.load(f))
            else:
                print(f"[load_experiment_results] File {filepath} does not exist.")
                walk_results.append(None)
        all_results.append(walk_results)
    return all_results

def run_and_save_experiment_samples_tesselation(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles,  # Fixed angles for all walks
    tesselation_orders_list_list,  # List of lists: [shift_prob][sample] -> tesselation_order for each walk
    initial_state_func,
    initial_state_kwargs,
    shift_probs,  # List of shift probabilities for each walk
    samples,  # Number of samples per shift probability
    base_dir="experiments_data_samples"
):
    """
    Run and save quantum walk experiments with multiple samples per tesselation shift probability.
    
    This function runs many quantum walks for different tesselation shift probabilities, 
    each with multiple samples, and saves the results to pickle files.
    
    Similar to run_and_save_experiment_samples but for tesselation order variations instead of angle variations.
    """
    print(f"[run_and_save_experiment] Starting tesselation sample experiment with {len(shift_probs)} shift probabilities and {samples} samples per shift probability")
    
    total_samples = len(shift_probs) * samples
    completed_samples = 0
    skipped_samples = 0
    computed_samples = 0
    total_start_time = time.time()
    
    for shift_prob_idx, shift_prob in enumerate(shift_probs):
        print(f"[run_and_save_experiment] Saving results to {get_experiment_dir(tesselation_func, shift_prob > 0, N, noise_params=[shift_prob] if shift_prob > 0 else [0], noise_type='tesselation_order', base_dir=base_dir)} for prob={shift_prob:.3f}")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        
        for sample_idx in range(samples):
            # Check if this sample already exists
            has_noise = shift_prob > 0
            noise_params = [shift_prob] if has_noise else [0]
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type="tesselation_order", base_dir=base_dir)
            
            # Create experiment directory
            os.makedirs(exp_dir, exist_ok=True)
            
            # Check if all step files for this sample exist
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                skipped_samples += 1
                completed_samples += 1
                elapsed_total = time.time() - total_start_time
                remaining_samples = total_samples - completed_samples
                if computed_samples > 0:
                    avg_time_per_computed_sample = elapsed_total / (completed_samples - skipped_samples)
                    eta_seconds = avg_time_per_computed_sample * (remaining_samples - skipped_samples)
                    print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped")
                    if eta_seconds > 0:
                        print(f"[ETA] Estimated time remaining: {eta_seconds/60:.1f} minutes ({eta_seconds:.1f} seconds)")
                else:
                    print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped")
                continue
            
            tesselation_order = tesselation_orders_list_list[shift_prob_idx][sample_idx]
            
            print(f"[run_and_save_experiment] Running walk for prob={shift_prob:.3f}, sample={sample_idx+1}/{samples}...")
            
            # Time each sample execution
            sample_start_time = time.time()
            
            # Build graph and initial state
            G = graph_func(N)
            initial_state = initial_state_func(N, **initial_state_kwargs)
            T = tesselation_func(N)
            
            final_states = running(
                G, T, steps,
                initial_state,
                angles=angles,
                tesselation_order=tesselation_order
            )
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            
            print(f"[run_and_save_experiment] Sample {sample_idx+1}/{samples} completed in {sample_duration:.2f} seconds")
            
            # Calculate progress and ETA
            completed_samples += 1
            computed_samples += 1
            dev_computed_samples += 1
            elapsed_total = time.time() - total_start_time
            remaining_samples = total_samples - completed_samples
            
            if computed_samples > 0:
                avg_time_per_computed_sample = elapsed_total / computed_samples
                eta_seconds = avg_time_per_computed_sample * (remaining_samples - skipped_samples)
                print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped")
                print(f"[ETA] Estimated time remaining: {eta_seconds/60:.1f} minutes ({eta_seconds:.1f} seconds)")
            
            # Save results for each time step
            for step_idx, state in enumerate(final_states):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                os.makedirs(step_dir, exist_ok=True)
                
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                with open(filepath, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        dev_end_time = time.time()
        dev_duration = dev_end_time - dev_start_time
        print(f"[run_and_save_experiment] Shift probability {shift_prob:.3f} completed in {dev_duration:.2f} seconds ({dev_computed_samples} samples computed)")
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"[run_and_save_experiment] All tesselation experiments completed in {total_duration:.2f} seconds")
    print(f"[run_and_save_experiment] Total: {computed_samples} samples computed, {skipped_samples} samples skipped")

def run_and_save_experiment_samples(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles_list_list,  # List of lists: [dev][sample] -> angles for each walk
    tesselation_order,
    initial_state_func,
    initial_state_kwargs,
    devs,  # List of devs for each walk
    samples,  # Number of samples per deviation
    base_dir="experiments_data_samples"
):
    """
    Runs the experiment for each dev with multiple samples and saves each sample's final states 
    in step folders within each dev folder. Includes progress tracking and memory optimization.
    """
    results = []
    total_start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0
    skipped_samples = 0
    computed_samples = 0
    
    for dev_idx, dev in enumerate(devs):
        has_noise = dev > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[run_and_save_experiment] Saving results to {exp_dir} for angle_dev={dev:.3f}")

        G = graph_func(N)
        T = tesselation_func(N)
        initial_state = initial_state_func(N, **initial_state_kwargs)

        # Track timing for this deviation
        dev_start_time = time.time()
        dev_results = []
        dev_computed_samples = 0
        dev_skipped_samples = 0
        
        for sample_idx in range(samples):
            # Check if this sample already exists - skip if it does
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                print(f"[run_and_save_experiment] Sample {sample_idx+1}/{samples} for dev={dev:.3f} already exists, skipping...")
                completed_samples += 1
                skipped_samples += 1
                dev_skipped_samples += 1
                
                # Update progress for skipped sample
                elapsed_total = time.time() - total_start_time
                remaining_samples = total_samples - completed_samples
                if completed_samples > skipped_samples:  # Only calculate ETA if we have computed samples
                    avg_time_per_computed_sample = elapsed_total / (completed_samples - skipped_samples)
                    eta_seconds = avg_time_per_computed_sample * (remaining_samples - skipped_samples)
                    print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped")
                    if eta_seconds > 0:
                        print(f"[ETA] Estimated time remaining: {eta_seconds/60:.1f} minutes ({eta_seconds:.1f} seconds)")
                else:
                    print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped")
                continue
            
            angles = angles_list_list[dev_idx][sample_idx]
            
            print(f"[run_and_save_experiment] Running walk for dev={dev:.3f}, sample={sample_idx+1}/{samples}...")
            
            # Time each sample execution
            sample_start_time = time.time()
            final_states = running(
                G, T, steps,
                initial_state,
                angles=angles,
                tesselation_order=tesselation_order
            )
            sample_end_time = time.time()
            sample_duration = sample_end_time - sample_start_time
            
            print(f"[run_and_save_experiment] Sample {sample_idx+1}/{samples} completed in {sample_duration:.2f} seconds")
            
            # Calculate progress and ETA
            completed_samples += 1
            computed_samples += 1
            dev_computed_samples += 1
            elapsed_total = time.time() - total_start_time
            avg_time_per_computed_sample = elapsed_total / computed_samples
            remaining_computed_samples = (total_samples - completed_samples) - (skipped_samples * (total_samples - completed_samples) / total_samples)
            eta_seconds = avg_time_per_computed_sample * remaining_computed_samples
            print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%) - {skipped_samples} skipped, {computed_samples} computed")
            if remaining_computed_samples > 0:
                print(f"[ETA] Estimated time remaining: {eta_seconds/60:.1f} minutes ({eta_seconds:.1f} seconds)")
            
            # Save each step's final state in its own step folder (optimized I/O)
            for step_idx, state in enumerate(final_states):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                os.makedirs(step_dir, exist_ok=True)
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                # Use highest pickle protocol for faster I/O
                with open(filepath, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Do not append final_states to dev_results; just save to file and free memory
            print(f"[run_and_save_experiment] Saved {len(final_states)} states for dev={dev:.3f}, sample={sample_idx}.")
            final_states = None
        
        # Summary for this deviation
        dev_total_time = time.time() - dev_start_time
        print(f"[Dev Summary] Completed all {samples} samples for dev={dev:.3f} in {dev_total_time:.2f} seconds")
        if dev_computed_samples > 0:
            print(f"[Dev Summary] Computed {dev_computed_samples} samples, skipped {dev_skipped_samples} existing samples")
            print(f"[Dev Summary] Average time per computed sample for dev={dev:.3f}: {dev_total_time/dev_computed_samples:.2f} seconds")
        else:
            print(f"[Dev Summary] All {samples} samples were already computed (skipped)")
        print("=" * 60)
        
        results.append(dev_results)
    
    # Final summary
    print(f"\n=== Experiment Completion Summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Computed samples: {computed_samples}")
    print(f"Skipped existing samples: {skipped_samples}")
    print(f"Computation efficiency: {computed_samples/total_samples*100:.1f}% new work")
    
    return results

def create_mean_probability_distributions(
    tesselation_func,
    N,
    steps,
    devs,
    samples,
    source_base_dir="experiments_data_samples",
    target_base_dir="experiments_data_samples_probDist",
    noise_type="angle"
):
    """
    Convert each sample to probability distribution and create mean probability distributions
    for each step, saving them to a new folder structure.
    """
    print(f"Creating mean probability distributions for {len(devs)} devs, {steps} steps, {samples} samples each...")
    total_start_time = time.time()
    
    for dev_idx, dev in enumerate(devs):
        dev_start_time = time.time()
        has_noise = dev > 0
        if noise_type == "angle":
            noise_params = [dev, dev]
            param_name = "angle_dev"
        else:  # tesselation_order
            noise_params = [dev]
            param_name = "prob"
        
        source_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=source_base_dir)
        target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=target_base_dir)
        
        os.makedirs(target_exp_dir, exist_ok=True)
        print(f"  Dev {dev_idx+1}/{len(devs)} ({param_name}={dev:.3f}): Processing {steps} steps...")
        
        for step_idx in range(steps):
            if step_idx % 50 == 0 or step_idx == steps - 1:
                elapsed = time.time() - dev_start_time
                progress = (step_idx + 1) / steps
                eta = elapsed / progress - elapsed if progress > 0 else 0
                print(f"    Step {step_idx+1}/{steps} ({progress*100:.1f}%) - ETA: {eta:.1f}s")
                
            step_dir = os.path.join(source_exp_dir, f"step_{step_idx}")
            
            # Load all samples for this step with optimized memory usage
            sample_states = []
            valid_samples = 0
            for sample_idx in range(samples):
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)
                    sample_states.append(state)
                    valid_samples += 1
                else:
                    print(f"Warning: Sample file not found: {filepath}")
            
            if sample_states and valid_samples > 0:
                # Convert quantum states to probability distributions with memory optimization
                prob_distributions = []
                for i, state in enumerate(sample_states):
                    prob_dist = amp2prob(state)  # |amplitude|²
                    prob_distributions.append(prob_dist)
                    # Clear state from memory to save RAM
                    sample_states[i] = None
                
                # Calculate mean probability distribution across samples
                mean_prob_dist = np.mean(prob_distributions, axis=0)
                
                # Clear prob_distributions to save memory
                del prob_distributions
                
                # Save mean probability distribution with high protocol
                mean_filename = f"mean_step_{step_idx}.pkl"
                mean_filepath = os.path.join(target_exp_dir, mean_filename)
                with open(mean_filepath, "wb") as f:
                    pickle.dump(mean_prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print(f"    No valid samples found for step {step_idx}")
        
        dev_time = time.time() - dev_start_time
        print(f"    Dev {dev_idx+1} completed in {dev_time:.1f}s")
    
    total_time = time.time() - total_start_time
    print(f"All mean probability distributions created in {total_time:.1f}s")

def load_mean_probability_distributions(
    tesselation_func,
    N,
    steps,
    devs,
    base_dir="experiments_data_samples_probDist",
    noise_type="angle"
):
    """
    Load the mean probability distributions from the probDist folder.
    Returns: List[List] - [dev][step] -> mean_probability_distribution
    """
    print(f"Loading mean probability distributions for {len(devs)} devs, {steps} steps each...")
    start_time = time.time()
    
    results = []
    for dev_idx, dev in enumerate(devs):
        dev_start_time = time.time()
        has_noise = dev > 0
        if noise_type == "angle":
            noise_params = [dev, dev]
            param_name = "angle_dev"
        else:  # tesselation_order
            noise_params = [dev]
            param_name = "prob"
            
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        
        print(f"  Dev {dev_idx+1}/{len(devs)} ({param_name}={dev:.3f}): Loading from {exp_dir}")
        
        dev_results = []
        for step_idx in range(steps):
            if step_idx % 100 == 0 or step_idx == steps - 1:
                print(f"    Loading step {step_idx+1}/{steps}...")
                
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                # Use optimized pickle loading
                with open(mean_filepath, "rb") as f:
                    mean_state = pickle.load(f)
                dev_results.append(mean_state)
            else:
                print(f"Warning: Mean probability distribution file not found: {mean_filepath}")
                dev_results.append(None)
        
        dev_time = time.time() - dev_start_time
        print(f"    Dev {dev_idx+1} completed in {dev_time:.1f}s ({len(dev_results)} steps loaded)")
        results.append(dev_results)
    
    total_time = time.time() - start_time
    print(f"All mean probability distributions loaded in {total_time:.1f}s")
    return results

def check_mean_probability_distributions_exist(
    tesselation_func,
    N,
    steps,
    devs,
    base_dir="experiments_data_samples_probDist",
    noise_type="angle"
):
    """
    Check if all mean probability distribution files exist.
    Returns: bool
    """
    print(f"Checking if mean probability distributions exist for {len(devs)} devs, {steps} steps each...")
    
    for dev_idx, dev in enumerate(devs):
        has_noise = dev > 0
        if noise_type == "angle":
            noise_params = [dev, dev]
            param_name = "angle_dev"
        else:  # tesselation_order
            noise_params = [dev]
            param_name = "prob"
            
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        
        print(f"  Checking dev {dev_idx+1}/{len(devs)} ({param_name}={dev:.3f}): {exp_dir}")
        
        missing_files = []
        for step_idx in range(steps):
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(exp_dir, mean_filename)
            if not os.path.exists(mean_filepath):
                missing_files.append(step_idx)
                
        if missing_files:
            print(f"    Missing {len(missing_files)} files (steps: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''})")
            return False
        else:
            print(f"    All {steps} files found!")
            
    print("All mean probability distribution files exist!")
    return True

def smart_load_or_create_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles_or_angles_list,  # Either fixed angles or list of angles/angles_list_list
    tesselation_order_or_list,  # Either fixed tesselation_order or list
    initial_state_func,
    initial_state_kwargs,
    parameter_list,  # List of parameter values (devs, shift_probs, etc.)
    samples=None,  # Number of samples (None for single walks)
    noise_type="angle",  # "angle" or "tesselation_order"
    parameter_name="param",  # Name for logging
    samples_base_dir="experiments_data_samples",
    probdist_base_dir="experiments_data_samples_probDist"
):
    """
    Intelligent loading function that follows the hierarchy:
    1. Try to load mean probability distributions (fastest)
    2. If not available, try to load samples and create probabilities
    3. If samples not available, run new experiment
    
    Returns: mean_probability_distributions [parameter][step] -> probability_distribution
    """
    print(f"Smart loading for {len(parameter_list)} {parameter_name} values...")
    start_time = time.time()
    
    # Step 1: Try to load mean probability distributions
    print("Step 1: Checking for existing mean probability distributions...")
    if check_mean_probability_distributions_exist(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type):
        print("✅ Found existing mean probability distributions - loading directly!")
        result = load_mean_probability_distributions(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type)
        elapsed = time.time() - start_time
        print(f"Smart loading completed in {elapsed:.1f}s (probability distributions path)")
        return result
    
    # Step 2: Try to load samples and create probability distributions
    if samples is not None:
        print("Step 2: Checking for existing sample data...")
        
        # Check if sample files exist
        sample_files_exist = True
        for param in parameter_list:
            has_noise = param > 0 if noise_type == "angle" else param > 0
            noise_params = [param, param] if noise_type == "angle" else [param]
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=samples_base_dir)
            
            for sample_idx in range(samples):
                for step_idx in range(steps):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    if not os.path.exists(filepath):
                        sample_files_exist = False
                        break
                if not sample_files_exist:
                    break
            if not sample_files_exist:
                break
        
        if sample_files_exist:
            print("✅ Found existing sample data - creating probability distributions...")
            create_mean_probability_distributions(tesselation_func, N, steps, parameter_list, samples, samples_base_dir, probdist_base_dir, noise_type)
            result = load_mean_probability_distributions(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type)
            elapsed = time.time() - start_time
            print(f"Smart loading completed in {elapsed:.1f}s (samples → probabilities path)")
            return result
    
    # Step 3: No existing data found - run new experiment
    print("Step 3: No existing data found - running new experiment...")
    
    if samples is not None:
        # Sample-based experiment
        print(f"Running sample-based experiment with {samples} samples per {parameter_name}...")
        
        # Convert parameters to the format expected by the sample functions
        if noise_type == "angle":
            # For angle experiments, we need angles_list_list
            if not isinstance(angles_or_angles_list[0], list) or not isinstance(angles_or_angles_list[0][0], list):
                # Convert single angles to samples format
                angles_list_list = []
                for param in parameter_list:
                    param_samples = []
                    for sample_idx in range(samples):
                        if param == 0:
                            param_samples.append(angles_or_angles_list)
                        else:
                            # Generate random angles for this sample
                            from sqw.utils import random_angle_deviation
                            param_samples.append(random_angle_deviation(angles_or_angles_list[0], [param, param], steps))
                    angles_list_list.append(param_samples)
            else:
                angles_list_list = angles_or_angles_list
                
            run_and_save_experiment_samples(
                graph_func, tesselation_func, N, steps, angles_list_list,
                tesselation_order_or_list, initial_state_func, initial_state_kwargs,
                parameter_list, samples, samples_base_dir
            )
        else:
            # For tesselation experiments with samples
            if not isinstance(tesselation_order_or_list[0], list) or not isinstance(tesselation_order_or_list[0][0], list):
                # Convert single tesselation to samples format
                tesselation_orders_list_list = []
                for param in parameter_list:
                    param_samples = []
                    for sample_idx in range(samples):
                        if param == 0:
                            param_samples.append(tesselation_order_or_list)
                        else:
                            # Generate random tesselation order for this sample
                            from sqw.utils import tesselation_choice
                            param_samples.append(tesselation_choice([[0, 1], [1, 0]], steps, [1 - param, param]))
                    tesselation_orders_list_list.append(param_samples)
            else:
                tesselation_orders_list_list = tesselation_order_or_list
                
            run_and_save_experiment_samples_tesselation(
                graph_func, tesselation_func, N, steps, angles_or_angles_list,
                tesselation_orders_list_list, initial_state_func, initial_state_kwargs,
                parameter_list, samples, samples_base_dir
            )
    
    if samples is None:
        # Single walk experiment 
        print("Running single-walk experiment...")
        noise_params_list = []
        for param in parameter_list:
            if noise_type == "angle":
                noise_params_list.append([param, param] if param > 0 else [0, 0])
            else:
                noise_params_list.append([param] if param > 0 else [0])
        
        run_and_save_experiment_generic(
            graph_func=graph_func, tesselation_func=tesselation_func, N=N, steps=steps,
            parameter_list=parameter_list, angles_or_angles_list=angles_or_angles_list,
            tesselation_order_or_list=tesselation_order_or_list,
            initial_state_func=initial_state_func, initial_state_kwargs=initial_state_kwargs,
            noise_params_list=noise_params_list, noise_type=noise_type, parameter_name=parameter_name,
            base_dir=samples_base_dir
        )
        
        # For single walks, convert to probability distributions manually
        print("Converting single walks to probability distributions...")
        results = load_experiment_results_generic(tesselation_func, N, steps, parameter_list, noise_params_list, noise_type, samples_base_dir)
        
        # Convert states to probability distributions and save
        os.makedirs(probdist_base_dir, exist_ok=True)
        
        for param_idx, param in enumerate(parameter_list):
            has_noise = param > 0 if noise_type == "angle" else param > 0
            noise_params = noise_params_list[param_idx]
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=probdist_base_dir)
            os.makedirs(target_exp_dir, exist_ok=True)
            
            walk_states = results[param_idx]
            for step_idx, state in enumerate(walk_states):
                if state is not None:
                    prob_dist = amp2prob(state)
                    mean_filename = f"mean_step_{step_idx}.pkl"
                    mean_filepath = os.path.join(target_exp_dir, mean_filename)
                    with open(mean_filepath, "wb") as f:
                        pickle.dump(prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # After creating data, load the probability distributions
    if samples is not None:
        create_mean_probability_distributions(tesselation_func, N, steps, parameter_list, samples, samples_base_dir, probdist_base_dir, noise_type)
    
    result = load_mean_probability_distributions(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type)
    elapsed = time.time() - start_time
    print(f"Smart loading completed in {elapsed:.1f}s (new experiment path)")
    return result
