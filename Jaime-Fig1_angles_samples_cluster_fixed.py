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
    required_modules = ["numpy", "scipy", "matplotlib", "networkx"]
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules

def zip_results(results_dir="experiments_data_samples", probdist_dir="experiments_data_samples_probDist"):
    """Bundle the results directories using native Linux tar (no compression)."""
    dirs_to_bundle = []
    
    if os.path.exists(results_dir):
        dirs_to_bundle.append(results_dir)
        print(f"Found results directory: {results_dir}")
    
    if os.path.exists(probdist_dir):
        dirs_to_bundle.append(probdist_dir)
        print(f"Found probability distributions directory: {probdist_dir}")
    
    if dirs_to_bundle:
        archive_filename = "quantum_walk_results.tar"
        
        try:
            # Use tar without compression (faster bundling, still single file)
            tar_cmd = f"tar -cf {archive_filename} " + " ".join(dirs_to_bundle)
            run_command(tar_cmd, check=False)
            if os.path.exists(archive_filename):
                print(f"Results bundled to {archive_filename}")
                return
        except:
            print("tar command failed, trying Python tarfile...")
            pass
        
        try:
            # Fallback to Python tarfile module without compression
            with tarfile.open(archive_filename, 'w') as tar:
                for dir_name in dirs_to_bundle:
                    tar.add(dir_name, arcname=os.path.basename(dir_name))
            print(f"Results bundled to {archive_filename}")
        except Exception as e:
            print(f"Warning: Could not create bundle: {e}")
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
        from sqw.tesselations import even_cycle_two_tesselation, even_line_two_tesselation
        from sqw.experiments_expanded import running, hamiltonian_builder, unitary_builder
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.statistics import states2mean, states2std, states2ipr, states2survival
        from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
        from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice
        
        try:
            from utils.plotTools import plot_qwak
        except ImportError:
            print("Warning: Could not import utils.plotTools, plotting will be disabled")
            
        from jaime_scripts import (
            get_experiment_dir, 
            run_and_save_experiment_generic, 
            load_experiment_results_generic,
            load_or_create_experiment_generic,
            plot_multiple_timesteps_qwak,
            plot_std_vs_time_qwak,
            plot_single_timestep_qwak
        )
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        sys.exit(1)
    
    import networkx as nx
    import numpy as np
    import os
    import pickle

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
        for dev_idx, dev in enumerate(devs):
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            os.makedirs(exp_dir, exist_ok=True)
            print(f"[run_and_save_experiment] Saving results to {exp_dir} for angle_dev={dev:.3f}")

            G = graph_func(N)
            T = tesselation_func(N)
            initial_state = initial_state_func(N, **initial_state_kwargs)

            # Run for each sample
            dev_results = []
            for sample_idx in range(samples):
                angles = angles_list_list[dev_idx][sample_idx]
                
                print(f"[run_and_save_experiment] Running walk for dev={dev:.3f}, sample={sample_idx}...")
                final_states = running(
                    G, T, steps,
                    initial_state,
                    angles=angles,
                    tesselation_order=tesselation_order
                )
                
                # Save each step's final state in its own step folder
                for step_idx, state in enumerate(final_states):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    os.makedirs(step_dir, exist_ok=True)
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    with open(os.path.join(step_dir, filename), "wb") as f:
                        pickle.dump(state, f)
                
                dev_results.append(final_states)
                print(f"[run_and_save_experiment] Saved {len(final_states)} states for dev={dev:.3f}, sample={sample_idx}.")
            
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
                
                # Load all samples for this step
                sample_states = []
                for sample_idx in range(samples):
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    
                    if os.path.exists(filepath):
                        with open(filepath, "rb") as f:
                            state = pickle.load(f)
                        sample_states.append(state)
                    else:
                        print(f"Warning: Sample file not found: {filepath}")
                
                if sample_states:
                    # Convert quantum states to probability distributions
                    prob_distributions = []
                    for state in sample_states:
                        prob_dist = amp2prob(state)  # |amplitude|²
                        prob_distributions.append(prob_dist)
                    
                    # Calculate mean probability distribution across samples
                    mean_prob_dist = np.mean(prob_distributions, axis=0)
                    
                    # Save mean probability distribution in the target directory
                    mean_filename = f"mean_step_{step_idx}.pkl"
                    mean_filepath = os.path.join(target_exp_dir, mean_filename)
                    with open(mean_filepath, "wb") as f:
                        pickle.dump(mean_prob_dist, f)
                    print(f"  Saved mean probability distribution for step {step_idx}")
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

    # Run the experiment with the exact same parameters as the original
    print("Starting quantum walk experiment...")
    
    # Use cluster-appropriate parameters (larger N for cluster computing)
    N = 2000  # Larger system size for cluster
    steps = N//4
    samples = 10  # Number of samples per deviation
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # List of devs and corresponding angles_list_list
    devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5,(np.pi/3) * 2]
    angles_list_list = []  # [dev][sample] -> angles
    
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

    print(f"Got results for {len(results_list)} devs with {samples} samples each")

    # Create or load mean probability distributions
    print("Creating or loading mean probability distributions...")
    mean_results = load_or_create_mean_probability_distributions(
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        devs=devs,
        samples=samples,
        source_base_dir="experiments_data_samples",
        target_base_dir="experiments_data_samples_probDist"
    )

    # Calculate statistics for verification (but skip plotting on cluster)
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

    print("Experiment completed successfully!")
    print(f"Raw results saved in experiments_data_samples/")
    print(f"Mean probability distributions saved in experiments_data_samples_probDist/")
    
    # Create TAR archive of results
    print("Creating TAR archive of results...")
    zip_results("experiments_data_samples", "experiments_data_samples_probDist")
    
    print("=== Analysis Instructions ===")
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
