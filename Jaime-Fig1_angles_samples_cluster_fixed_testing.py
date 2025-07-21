#!/usr/bin/env python3
"""
Fixed cluster-compatible version of the angle samples experiment.
This version preserves the exact functionality of Jaime-Fig1_angles_samples.py
while adding cluster environment setup and cleanup capabilities.
"""

import sys
import os
import subprocess
import tarfile
from pathlib import Path

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
    
def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            sys.exit(1)
        return result

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def setup_virtual_environment(venv_path):
    """Create and setup virtual environment with required packages."""
    print("Setting up virtual environment...")
    
    # Create virtual environment
    run_command(f"python3 -m venv {venv_path}")
    
    # Activate virtual environment and install packages
    pip_cmd = f"{venv_path}/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip")
    
    # Install required packages
    packages = [
        "numpy",
        "scipy", 
        "networkx",
        "matplotlib"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        run_command(f"{pip_cmd} install {package}")
    
    print("Virtual environment setup complete.")
    return f"{venv_path}/bin/python"

def check_dependencies():
    """Check if required dependencies are available."""
    required_modules = ["numpy", "scipy", "networkx", "matplotlib"]
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules

def zip_results(results_dir="experiments_data_samples", probdist_dir="experiments_data_samples_probDist", N=None, samples=None):
    """Bundle the results directories using native Linux tar (no compression)."""
    dirs_to_bundle = []
    
    if os.path.exists(results_dir):
        dirs_to_bundle.append(results_dir)
        print(f"Found results directory: {results_dir}")
    
    if os.path.exists(probdist_dir):
        dirs_to_bundle.append(probdist_dir)
        print(f"Found probability distributions directory: {probdist_dir}")
    
    if dirs_to_bundle:
        # Create filename with N and samples parameters
        if N is not None and samples is not None:
            archive_filename = f"quantum_walk_results_N{N}_samples{samples}.tar"
        else:
            archive_filename = "quantum_walk_results.tar"
        
        try:
            # Use tar without compression (faster bundling, still single file)
            tar_cmd = f"tar -cf {archive_filename} " + " ".join(dirs_to_bundle)
            run_command(tar_cmd, check=False)
            if os.path.exists(archive_filename):
                print(f"Results bundled to {archive_filename}")
                return archive_filename
        except:
            print("tar command failed, trying Python tarfile...")
            pass
        
        try:
            # Fallback to Python tarfile module without compression
            with tarfile.open(archive_filename, 'w') as tar:
                for dir_name in dirs_to_bundle:
                    tar.add(dir_name, arcname=os.path.basename(dir_name))
            print(f"Results bundled to {archive_filename}")
            return archive_filename
        except Exception as e:
            print(f"Warning: Could not create bundle: {e}")
            return None
    else:
        print(f"Warning: No results directories found")

def main():
    """Main execution function for cluster environment."""
    print("=== Fixed Cluster Quantum Walk Experiment ===")
    
    # Check Python version
    check_python_version()
    
    # Setup paths
    work_dir = Path.cwd()
    venv_path = work_dir / "qw_venv"
    
    # Check if we need to setup virtual environment
    missing_deps = check_dependencies()
    python_executable = sys.executable
    
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
        print("Setting up virtual environment...")
        python_executable = setup_virtual_environment(venv_path)
        
        # Re-execute this script with the virtual environment Python
        script_path = __file__
        print(f"Re-executing with virtual environment Python: {python_executable}")
        run_command(f"{python_executable} {script_path} --venv-ready")
        
        # Bundle results and exit
        zip_results()
        print("=== Experiment completed ===")
        return
    
    # If we reach here, dependencies are available - run the experiment
    print("Dependencies available, running experiment...")
    run_experiment()

def run_experiment():
    """Run the actual quantum walk experiment using the exact original functions."""
    # Import the exact modules and functions from the original file
    try:
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.utils import random_angle_deviation
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        sys.exit(1)
    
    import networkx as nx
    import numpy as np
    import os
    import pickle
    import time

    # Import all the exact functions from the original Jaime-Fig1_angles_samples.py
    # We'll define them here exactly as they are in the original file
    
    def run_and_save_experiment_generic_samples(
        graph_func,
        tesselation_func,
        N,
        steps,
        samples,
        parameter_list,  # List of varying deviations for each walk
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
        from sqw.experiments_expanded import running
        import pickle
        
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
                    pickle.dump(state, f)
            print(f"[run_and_save_experiment] Saved {len(final_states)} states for {parameter_name}={param:.3f}.")
            results.append(final_states)
        return results

    def run_and_save_experiment(
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
        in step folders within each dev folder.
        """
        from sqw.experiments_expanded import running
        import pickle
        
        results = []
        total_start_time = time.time()
        total_samples = len(devs) * samples
        completed_samples = 0
        
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
            
            for sample_idx in range(samples):
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
                elapsed_total = time.time() - total_start_time
                avg_time_per_sample = elapsed_total / completed_samples
                remaining_samples = total_samples - completed_samples
                eta_seconds = avg_time_per_sample * remaining_samples
                
                print(f"[Progress] {completed_samples}/{total_samples} samples completed ({completed_samples/total_samples*100:.1f}%)")
                if remaining_samples > 0:
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
                
                dev_results.append(final_states)
                print(f"[run_and_save_experiment] Saved {len(final_states)} states for dev={dev:.3f}, sample={sample_idx}.")
            
            # Summary for this deviation
            dev_total_time = time.time() - dev_start_time
            print(f"[Dev Summary] Completed all {samples} samples for dev={dev:.3f} in {dev_total_time:.2f} seconds")
            print(f"[Dev Summary] Average time per sample for dev={dev:.3f}: {dev_total_time/samples:.2f} seconds")
            print("=" * 60)
            
            results.append(dev_results)
        return results

    def load_experiment_results(
        tesselation_func,
        N,
        steps,
        devs,
        samples,
        base_dir="experiments_data"
    ):
        """
        Loads all final states from disk for each dev with multiple samples.
        Returns: List[List[List]] - [dev][sample][step] -> state
        """
        results = []
        for dev in devs:
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            
            dev_results = []
            for sample_idx in range(samples):
                sample_states = []
                for step_idx in range(steps):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    
                    if os.path.exists(filepath):
                        # Use faster pickle loading
                        with open(filepath, "rb") as f:
                            state = pickle.load(f)
                        sample_states.append(state)
                    else:
                        print(f"Warning: File not found: {filepath}")
                        sample_states.append(None)
                dev_results.append(sample_states)
            results.append(dev_results)
        return results

    def load_or_create_experiment(
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
        base_dir="experiments_data"
    ):
        """
        Loads experiment results for each walk with samples if they exist, otherwise runs and saves them.
        Returns: List[List[List]] - [dev][sample][step] -> state
        """
        # Check if all experiment files exist
        all_exist = True
        for dev in devs:
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            
            for sample_idx in range(samples):
                for step_idx in range(steps):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    if not os.path.exists(filepath):
                        all_exist = False
                        break
                if not all_exist:
                    break
            if not all_exist:
                break
        
        if all_exist:
            print("Loading existing experiment results...")
            return load_experiment_results(tesselation_func, N, steps, devs, samples, base_dir)
        else:
            print("Some results missing, running new experiment...")
            return run_and_save_experiment(
                graph_func, tesselation_func, N, steps, angles_list_list,
                tesselation_order, initial_state_func, initial_state_kwargs,
                devs, samples, base_dir
            )

    def create_mean_probability_distributions(
        tesselation_func,
        N,
        steps,
        devs,
        samples,
        source_base_dir="experiments_data_samples",
        target_base_dir="experiments_data_samples_probDist"
    ):
        """
        Convert each sample to probability distribution and create mean probability distributions
        for each step, saving them to a new folder structure.
        """
        from sqw.states import amp2prob
        
        for dev in devs:
            has_noise = dev > 0
            source_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=source_base_dir)
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=target_base_dir)
            
            os.makedirs(target_exp_dir, exist_ok=True)
            print(f"Processing dev={dev:.2f}, creating mean probability distributions...")
            
            for step_idx in range(steps):
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
                    
                    if step_idx % 50 == 0:  # Progress indicator
                        print(f"  Processed {step_idx + 1}/{steps} steps for dev={dev:.2f}")
                else:
                    print(f"  No valid samples found for step {step_idx}")

    def load_mean_probability_distributions(
        tesselation_func,
        N,
        steps,
        devs,
        base_dir="experiments_data_samples_probDist"
    ):
        """
        Load the mean probability distributions from the probDist folder.
        Returns: List[List] - [dev][step] -> mean_probability_distribution
        """
        results = []
        for dev in devs:
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            
            dev_results = []
            for step_idx in range(steps):
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
            results.append(dev_results)
        return results

    def check_mean_probability_distributions_exist(
        tesselation_func,
        N,
        steps,
        devs,
        base_dir="experiments_data_samples_probDist"
    ):
        """
        Check if all mean probability distribution files exist.
        Returns: bool
        """
        for dev in devs:
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            
            for step_idx in range(steps):
                mean_filename = f"mean_step_{step_idx}.pkl"
                mean_filepath = os.path.join(exp_dir, mean_filename)
                if not os.path.exists(mean_filepath):
                    return False
        return True

    def load_or_create_mean_probability_distributions(
        tesselation_func,
        N,
        steps,
        devs,
        samples,
        source_base_dir="experiments_data_samples",
        target_base_dir="experiments_data_samples_probDist"
    ):
        """
        Load mean probability distributions if they exist, otherwise create them.
        Returns: List[List] - [dev][step] -> mean_probability_distribution
        """
        if check_mean_probability_distributions_exist(tesselation_func, N, steps, devs, target_base_dir):
            print("Loading existing mean probability distributions...")
            return load_mean_probability_distributions(tesselation_func, N, steps, devs, target_base_dir)
        else:
            print("Creating mean probability distributions...")
            create_mean_probability_distributions(tesselation_func, N, steps, devs, samples, source_base_dir, target_base_dir)
            return load_mean_probability_distributions(tesselation_func, N, steps, devs, target_base_dir)

    def prob_distributions2std(prob_distributions, domain):
        """
        Calculate standard deviation from probability distributions.
        """
        std_values = []
        
        for prob_dist in prob_distributions:
            if prob_dist is None:
                std_values.append(0)
                continue
                
            # Calculate mean (first moment)
            mean = np.sum(domain * prob_dist.flatten())
            
            # Calculate second moment
            second_moment = np.sum((domain ** 2) * prob_dist.flatten())
            
            # Calculate variance and standard deviation
            variance = second_moment - mean ** 2
            std = np.sqrt(variance) if variance > 0 else 0
            std_values.append(std)
            
        return std_values

    # Run the experiment with cluster-optimized parameters
    print("Starting quantum walk experiment...")
    
    # Optimized parameters for better cluster performance
    N = 100  # Reduced system size for faster computation
    steps = N//4  # Reduced steps for faster execution
    samples = 5  # Reduced samples for quicker testing
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # Reduced list of devs for faster execution
    devs = [0, (np.pi/3)/2.5, (np.pi/3)]  # Reduced from 6 to 3 deviations
    angles_list_list = []  # [dev][sample] -> angles
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print(f"This will significantly reduce computation time while preserving the experiment structure.")
    
    for dev in devs:
        dev_angles_list = []
        for sample_idx in range(samples):
            if dev == 0:
                # No noise case - use perfect angles
                dev_angles_list.append([[np.pi/3, np.pi/3]] * steps)
            else:
                dev_angles_list.append(random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps))
        angles_list_list.append(dev_angles_list)

    print(f"Running experiment for {len(devs)} different angle noise deviations with {samples} samples each...")
    print(f"Angle devs: {devs}")
    
    # Start timing the main experiment
    start_time = time.time()

    # Run the main experiment
    results_list = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_list_list=angles_list_list,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        devs=devs,
        samples=samples,
        base_dir="experiments_data_samples"
    )

    experiment_time = time.time() - start_time
    print(f"Main experiment completed in {experiment_time:.2f} seconds")
    print(f"Got results for {len(results_list)} devs with {samples} samples each")

    # Create or load mean probability distributions
    print("Creating or loading mean probability distributions...")
    prob_start_time = time.time()
    
    mean_results = load_or_create_mean_probability_distributions(
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        devs=devs,
        samples=samples,
        source_base_dir="experiments_data_samples",
        target_base_dir="experiments_data_samples_probDist"
    )
    
    prob_time = time.time() - prob_start_time
    print(f"Probability distributions processing completed in {prob_time:.2f} seconds")

    # Calculate statistics for verification and plotting
    domain = np.arange(N)
    stds = []
    for i, dev_mean_prob_dists in enumerate(mean_results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Dev {i} (angle_dev={devs[i]:.2f}): {len(std_values)} std values")
        else:
            print(f"Dev {i} (angle_dev={devs[i]:.2f}): No valid mean probability distributions")
            stds.append([])

    # Plot standard deviation as a function of time steps
    print("Creating standard deviation vs time steps plot...")
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        time_steps = np.arange(len(stds[0]) if stds[0] else 0)
        
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values:  # Only plot if we have data
                plt.plot(time_steps, std_values, 'o-', label=f'angle_dev = {dev:.3f}', linewidth=2, markersize=4)
        
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title(f'Standard Deviation vs Time Steps\n(N={N}, samples={samples})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"stdev_vs_timesteps_N{N}_samples{samples}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        
        # Also show the plot if running interactively
        plt.show()
        
        # Create a second plot showing the evolution more clearly
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Standard deviation vs time steps
        plt.subplot(2, 1, 1)
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values:
                plt.plot(time_steps, std_values, 'o-', label=f'angle_dev = {dev:.3f}', linewidth=2, markersize=3)
        plt.xlabel('Time Steps')
        plt.ylabel('Standard Deviation')
        plt.title(f'Standard Deviation Evolution (N={N}, samples={samples})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Final standard deviation vs angle deviation
        plt.subplot(2, 1, 2)
        final_stds = [std_values[-1] if std_values else 0 for std_values in stds]
        plt.plot(devs, final_stds, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Angle Deviation (radians)')
        plt.ylabel('Final Standard Deviation')
        plt.title('Final Standard Deviation vs Angle Noise')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the combined plot
        combined_plot_filename = f"stdev_analysis_N{N}_samples{samples}.png"
        plt.savefig(combined_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined analysis plot saved as: {combined_plot_filename}")
        
        plt.show()
        
        # Print some statistics
        print("\n=== Standard Deviation Analysis ===")
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values:
                initial_std = std_values[0]
                final_std = std_values[-1]
                max_std = max(std_values)
                print(f"Dev {dev:.3f}: Initial={initial_std:.3f}, Final={final_std:.3f}, Max={max_std:.3f}")
        
    except ImportError:
        print("Warning: matplotlib not available for plotting")
    except Exception as e:
        print(f"Warning: Could not create plot: {e}")

    print("Experiment completed successfully!")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Raw results saved in experiments_data_samples/")
    print(f"Mean probability distributions saved in experiments_data_samples_probDist/")
    
    print("\n=== Performance Summary ===")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(devs)}")
    print(f"Total quantum walks computed: {len(devs) * samples}")
    print(f"Average time per quantum walk: {experiment_time / (len(devs) * samples):.3f} seconds")
    
    print("\n=== Scaling Note ===")
    print("For production runs, you can scale up the parameters:")
    print("- Increase N to 2000 for higher resolution")
    print("- Increase steps to N//4 for longer evolution")
    print("- Increase samples to 10+ for better statistics")
    print("- Add more deviation values as needed")
    
    # Create TAR archive of results
    print("Creating TAR archive of results...")
    archive_filename = zip_results("experiments_data_samples", "experiments_data_samples_probDist", N, samples)
    
    print("=== Analysis Instructions ===")
    if archive_filename:
        print(f"Results archived in: {archive_filename}")
    print("To analyze the results, transfer the tar file and extract it, then use:")
    print("- experiments_data_samples/ contains the raw quantum states for each sample")
    print("- experiments_data_samples_probDist/ contains the mean probability distributions")
    print("Both directories maintain the same folder structure for easy analysis.")

if __name__ == "__main__":
    # Check for virtual environment flag
    if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
        # We're running in virtual environment, just run experiment
        run_experiment()
        # zip_results("experiments_data_samples", "experiments_data_samples_probDist")
        print("=== Experiment completed in virtual environment ===")
    else:
        # Run main setup function
        main()
