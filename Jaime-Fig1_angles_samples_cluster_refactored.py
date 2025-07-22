#!/usr/bin/env python3
"""
Refactored cluster-compatible version of the angle samples experiment.
This version uses shared functions from jaime_scripts to eliminate code duplication.
"""

import sys
import os
import subprocess
import tarfile
import time
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
                for dir_path in dirs_to_bundle:
                    tar.add(dir_path)
            print(f"Results bundled to {archive_filename}")
            return archive_filename
        except Exception as e:
            print(f"Warning: Could not create bundle: {e}")
            return None
    else:
        print(f"Warning: No results directories found")
        return None

def main():
    """Main execution function for cluster environment."""
    print("=== Refactored Cluster Quantum Walk Experiment ===")
    
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
    """Run the actual quantum walk experiment using shared functions."""
    # Import the exact modules and functions from the original file
    try:
        import numpy as np
        import networkx as nx
        import pickle
        
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.utils import random_angle_deviation
        
        # Import shared functions from jaime_scripts after dependencies are confirmed
        from jaime_scripts import (
            get_experiment_dir,
            run_and_save_experiment_samples,
            load_experiment_results_samples,
            load_or_create_experiment_samples,
            create_mean_probability_distributions,
            load_mean_probability_distributions,
            check_mean_probability_distributions_exist,
            load_or_create_mean_probability_distributions,
            prob_distributions2std,
            smart_load_or_create_experiment  # New intelligent loading function
        )
        
        print("Successfully imported all required modules")
    except ImportError as e:
        print(f"Error: Could not import required modules: {e}")
        print("Make sure you're running this script from the correct directory with all dependencies available")
        sys.exit(1)

    # Run the experiment with cluster-optimized parameters
    print("Starting quantum walk experiment...")
    
    # Optimized parameters for better cluster performance
    N = 2000  # System size
    steps = N//4  # Time steps
    samples = 10  # Samples per deviation
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # List of angle noise deviations
    devs = [0, (np.pi/3)/2.5, (np.pi/3)*2]
    angles_list_list = []  # [dev][sample] -> angles
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print(f"This will significantly reduce computation time while preserving the experiment structure.")
    
    # Generate angle sequences for each deviation and sample
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

    # Use the new smart loading function that follows the hierarchy:
    # 1. Try probability distributions first (fastest)
    # 2. Try samples if probabilities don't exist
    # 3. Create new experiment if nothing exists
    mean_results = smart_load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_or_angles_list=angles_list_list,
        tesselation_order_or_list=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=devs,
        samples=samples,
        noise_type="angle",
        parameter_name="angle_dev",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )

    experiment_time = time.time() - start_time
    print(f"Smart loading completed in {experiment_time:.2f} seconds")
    print(f"Processed {len(devs)} deviations with {samples} samples each")

    # Calculate statistics for verification (but skip plotting on cluster)
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
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(devs) * samples):.3f} seconds")
    
    print("\n=== Scaling Note ===")
    print("For production runs, you can scale up the parameters:")
    print("- Increase N for higher resolution")
    print("- Increase steps for longer evolution")
    print("- Increase samples for better statistics")
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
        print("=== Experiment completed in virtual environment ===")
    else:
        # Run main setup function
        main()
