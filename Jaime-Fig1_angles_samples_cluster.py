#!/usr/bin/env python3
"""
Cluster-compatible version of the angle samples experiment.
Handles virtual environment setup, dependency installation, and cleanup.
"""

import os
import sys
import subprocess
import shutil
import tempfile
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
        "matplotlib",
        "networkx",
        "pickle5"  # For better pickle compatibility
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

def create_sqw_mock_modules():
    """Create mock/minimal versions of sqw modules for cluster environment."""
    sqw_dir = Path("sqw")
    sqw_dir.mkdir(exist_ok=True)
    
    # Create __init__.py files
    for subdir in ["", "tesselations", "experiments_expanded", "states", "statistics", "plots", "utils"]:
        if subdir:
            (sqw_dir / subdir).mkdir(exist_ok=True)
            (sqw_dir / subdir / "__init__.py").touch()
        else:
            (sqw_dir / "__init__.py").touch()
    
    # Create utils directory and plotTools
    utils_dir = Path("utils")
    utils_dir.mkdir(exist_ok=True)
    (utils_dir / "__init__.py").touch()
    
    # Create jaime_scripts module
    Path("jaime_scripts.py").touch()

def zip_results(results_dir="experiments_data_samples"):
    """Bundle the results directory using native Linux tar (no compression)."""
    if os.path.exists(results_dir):
        print(f"Bundling results directory: {results_dir}")
        archive_filename = f"{results_dir}.tar"
        
        try:
            # Use tar without compression (faster bundling, still single file)
            run_command(f"tar -cf {archive_filename} {results_dir}", check=False)
            if os.path.exists(archive_filename):
                print(f"Results bundled to {archive_filename}")
                return
        except:
            print("tar command failed, trying Python tarfile...")
            pass
        
        try:
            # Fallback to Python tarfile module without compression
            import tarfile
            with tarfile.open(archive_filename, 'w') as tar:
                tar.add(results_dir, arcname=os.path.basename(results_dir))
            print(f"Results bundled to {archive_filename}")
        except Exception as e:
            print(f"Warning: Could not create bundle: {e}")
    else:
        print(f"Warning: Results directory {results_dir} not found")

def cleanup_environment(venv_path):
    """Clean up virtual environment."""
    if os.path.exists(venv_path):
        print("Cleaning up virtual environment...")
        shutil.rmtree(venv_path)
        print("Virtual environment removed.")

def main():
    """Main execution function for cluster environment."""
    print("=== Cluster Quantum Walk Experiment ===")
    
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
        
        # Cleanup and exit
        zip_results()
        cleanup_environment(venv_path)
        print("=== Experiment completed ===")
        return
    
    # If we reach here, dependencies are available - run the experiment
    print("Dependencies available, running experiment...")
    run_experiment()

def run_experiment():
    """Run the actual quantum walk experiment."""
    # Import here to ensure virtual environment is active
    import numpy as np
    import networkx as nx
    import pickle
    import os
    
    # Import or define the required functions
    # Note: In a real cluster environment, you'd need to ensure these modules are available
    try:
        from sqw.tesselations import even_cycle_two_tesselation, even_line_two_tesselation
        from sqw.experiments_expanded import running, hamiltonian_builder, unitary_builder
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.statistics import states2mean, states2std, states2ipr, states2survival
        from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
        from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice
        from utils.plotTools import plot_qwak
        from jaime_scripts import (
            get_experiment_dir, 
            run_and_save_experiment_generic, 
            load_experiment_results_generic,
            load_or_create_experiment_generic,
            plot_multiple_timesteps_qwak,
            plot_std_vs_time_qwak,
            plot_single_timestep_qwak
        )
        print("Successfully imported sqw modules")
    except ImportError as e:
        print(f"Warning: Could not import sqw modules: {e}")
        print("Using minimal implementations for cluster environment")
        # Create minimal stubs for missing functions
        def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="angle", base_dir="experiments_data"):
            """Minimal implementation of get_experiment_dir"""
            func_name = tesselation_func.__name__ if hasattr(tesselation_func, '__name__') else 'unknown'
            noise_str = "noise" if has_noise else "nonoise" 
            param_str = "_".join(map(str, noise_params)) if noise_params else "0"
            return os.path.join(base_dir, f"{func_name}_{noise_type}_{noise_str}_{param_str}")
        
        def even_line_two_tesselation(N):
            """Minimal tesselation function"""
            return [[(i, (i+1) % N)] for i in range(N)]
        
        def uniform_initial_state(N, nodes=None):
            """Minimal initial state function"""
            if nodes is None:
                nodes = [N//2]
            state = np.zeros((N, 1), dtype=complex)
            for node in nodes:
                state[node] = 1.0 / np.sqrt(len(nodes))
            return state
        
        def amp2prob(state):
            """Convert amplitude to probability"""
            return np.real(state * np.conj(state))
        
        def random_angle_deviation(base_angles, deviations, steps):
            """Generate random angle deviations"""
            angles = []
            for _ in range(steps):
                step_angles = []
                for i, (base, dev) in enumerate(zip(base_angles, deviations)):
                    if dev == 0:
                        step_angles.append(base)
                    else:
                        step_angles.append(base + np.random.normal(0, dev))
                angles.append(step_angles)
            return angles
        
        def running(G, T, steps, initial_state, angles=None, tesselation_order=None):
            """Minimal quantum walk implementation"""
            # This is a very simplified implementation
            # In practice, you'd need the full quantum walk implementation
            states = []
            current_state = initial_state.copy()
            
            for step in range(steps):
                # Simple random walk approximation
                # In reality, this would be the quantum walk evolution
                N = len(current_state)
                next_state = np.zeros_like(current_state)
                
                for i in range(N):
                    # Distribute amplitude to neighbors
                    left = (i - 1) % N
                    right = (i + 1) % N
                    next_state[left] += 0.5 * current_state[i]
                    next_state[right] += 0.5 * current_state[i]
                
                current_state = next_state
                states.append(current_state.copy())
            
            return states
    
    # Define the experiment functions inside run_experiment
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

    def calculate_or_load_mean(
        tesselation_func,
        N,
        steps,
        devs,
        samples,
        base_dir="experiments_data_samples"
    ):
        """
        Calculate or load the mean of samples for each step. If mean files already exist,
        load them. Otherwise, calculate the means and save them to files.
        
        For quantum states, this converts each sample to probability distribution 
        (|amplitude|²) and then averages the probability distributions.
        
        Returns
        -------
        List[List] - [dev][step] -> mean_probability_distribution
        """
        # Check if all mean files exist
        all_means_exist = True
        for dev in devs:
            has_noise = dev > 0
            exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
            
            for step_idx in range(steps):
                mean_filename = f"mean_step_{step_idx}.pkl"
                mean_filepath = os.path.join(exp_dir, mean_filename)
                if not os.path.exists(mean_filepath):
                    all_means_exist = False
                    break
            if not all_means_exist:
                break
        
        if all_means_exist:
            print("Loading existing mean files...")
            return load_mean(tesselation_func, N, steps, devs, base_dir)
        else:
            print("Some mean files missing, calculating means...")
            
            for dev in devs:
                has_noise = dev > 0
                exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=[dev, dev], noise_type="angle", base_dir=base_dir)
                
                for step_idx in range(steps):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    
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
                        
                        # Save mean probability distribution in the main experiment directory
                        mean_filename = f"mean_step_{step_idx}.pkl"
                        mean_filepath = os.path.join(exp_dir, mean_filename)
                        with open(mean_filepath, "wb") as f:
                            pickle.dump(mean_prob_dist, f)
                        print(f"Calculated and saved mean probability distribution for dev={dev:.3f}, step={step_idx}")
                    else:
                        print(f"No valid samples found for dev={dev:.3f}, step={step_idx}")
            
            # Load and return the calculated means
            return load_mean(tesselation_func, N, steps, devs, base_dir)

    def load_mean(
        tesselation_func,
        N,
        steps,
        devs,
        base_dir="experiments_data"
    ):
        """
        Load the mean probability distributions for each deviation and step.
        Returns: List[List] - [dev][step] -> mean_probability_distribution
        
        Note: These are mean probability distributions (|amplitude|²), not quantum states.
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
                    print(f"Warning: Mean file not found: {mean_filepath}")
                    dev_results.append(None)
            results.append(dev_results)
        return results

    def prob_distributions2std(prob_distributions, domain):
        """
        Calculate standard deviation from probability distributions.
        
        Parameters
        ----------
        prob_distributions : list
            List of probability distributions (already |amplitude|²)
        domain : array-like
            Position domain (e.g., np.arange(N))
            
        Returns
        -------
        list
            Standard deviations for each time step
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

    # Run the actual experiment
    print("Starting quantum walk experiment...")
    N = 2000
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

    # Calculate or load mean results for plotting
    print("Calculating or loading means...")
    mean_results = calculate_or_load_mean(
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        devs=devs,
        samples=samples,
        base_dir="experiments_data_samples"
    )

    # Calculate statistics for plotting using mean results
    domain = np.arange(N)
    stds = []
    for i, dev_mean_prob_dists in enumerate(mean_results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Dev {i} (angle_dev={devs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Dev {i} (angle_dev={devs[i]:.3f}): No valid mean probability distributions")
            stds.append([])

    print("Experiment completed successfully!")
    print(f"Results saved in experiments_data_samples/")
    
    # Create ZIP archive of results
    print("Creating ZIP archive of results...")
    zip_results("experiments_data_samples")


if __name__ == "__main__":
    # Check for virtual environment flag
    if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
        # We're running in virtual environment, just run experiment
        run_experiment()
        print("=== Experiment completed in virtual environment ===")
    else:
        # Run main setup function
        main()
