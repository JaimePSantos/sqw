#!/usr/bin/env python3
"""
Cluster-compatible version of the tesselation order samples experiment.
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
        "matplotlib",
        "qwak-sim"  # Assuming qwak-sim is the package for quantum walk simulations
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
            archive_filename = f"tesselation_results_N{N}_samples{samples}.tar"
        else:
            archive_filename = "tesselation_results.tar"
        
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
    print("=== Cluster Tesselation Order Quantum Walk Experiment ===")
    
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
        print("=== Tesselation Experiment completed ===")
        return
    
    # If we reach here, dependencies are available - run the experiment
    print("Dependencies available, running tesselation experiment...")
    run_experiment()

def run_experiment():
    """Run the actual tesselation quantum walk experiment using shared functions."""
    # Import the exact modules and functions from the original file
    try:
        import numpy as np
        import networkx as nx
        import pickle
        
        from sqw.tesselations import even_line_two_tesselation
        from sqw.experiments_expanded import running
        from sqw.states import uniform_initial_state, amp2prob
        from sqw.utils import tesselation_choice
        
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

    # Run the tesselation experiment with cluster-optimized parameters
    print("Starting tesselation quantum walk experiment...")
    
    # Optimized parameters for better cluster performance
    N = 2000  # System size
    steps = N//4  # Time steps
    samples = 10  # Samples per shift probability
    angles = [[np.pi/3, np.pi/3]] * steps  # Fixed angles, no noise
    initial_state_kwargs = {"nodes": [N//2]}

    # List of tesselation shift probabilities
    shift_probs = [0, 0.2, 0.5]
    tesselation_orders_list = []  # [shift_prob][sample] -> tesselation_order
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print(f"This will significantly reduce computation time while preserving the experiment structure.")
    
    # Generate tesselation orders for each shift probability and sample
    for shift_prob in shift_probs:
        shift_tesselation_orders = []
        for sample_idx in range(samples):
            if shift_prob == 0:
                # No noise case - use fixed tesselation order
                shift_tesselation_orders.append([[0, 1]] * steps)
            else:
                # Noisy tesselation order - generate different random sequence for each sample
                tesselation_order = tesselation_choice([[0, 1], [1, 0]], steps, [1 - shift_prob, shift_prob])
                shift_tesselation_orders.append(tesselation_order)
        tesselation_orders_list.append(shift_tesselation_orders)

    print(f"Running experiment for {len(shift_probs)} different tesselation shift probabilities with {samples} samples each...")
    print(f"Shift probabilities: {shift_probs}")
    
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
        angles_or_angles_list=angles,  # Fixed angles for all experiments
        tesselation_order_or_list=tesselation_orders_list,  # 2D list [shift_prob][sample] -> tesselation_order
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=shift_probs,
        samples=samples,
        noise_type="tesselation_order",
        parameter_name="prob",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )

    experiment_time = time.time() - start_time
    print(f"Smart loading completed in {experiment_time:.2f} seconds")
    print(f"Processed {len(shift_probs)} shift probabilities with {samples} samples each")

    # Calculate statistics for verification (but skip plotting on cluster)
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

    print("Tesselation experiment completed successfully!")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Raw results saved in experiments_data_samples/")
    print(f"Mean probability distributions saved in experiments_data_samples_probDist/")
    
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
    
    print("\n=== Scaling Note ===")
    print("For production runs, you can scale up the parameters:")
    print("- Increase N for higher resolution")
    print("- Increase steps for longer evolution")
    print("- Increase samples for better statistics")
    print("- Add more shift probability values as needed")
    print("- Adjust shift probabilities to explore different noise regimes")
    
    # Create TAR archive of results
    print("Creating TAR archive of results...")
    archive_filename = zip_results("experiments_data_samples", "experiments_data_samples_probDist", N, samples)
    
    print("=== Analysis Instructions ===")
    if archive_filename:
        print(f"Results archived in: {archive_filename}")
    print("To analyze the results, transfer the tar file and extract it, then use:")
    print("- experiments_data_samples/ contains the raw quantum states for each sample")
    print("- experiments_data_samples_probDist/ contains the mean probability distributions")
    print("Both directories maintain the tesselation_order directory structure for easy analysis.")
    print("\nDirectory structure:")
    print("experiments_data_samples/even_line_two_tesselation_tesselation_order_nonoise/N_2000/")
    print("experiments_data_samples/even_line_two_tesselation_tesselation_order_noise/tesselation_shift_prob_X.XXX/N_2000/")

if __name__ == "__main__":
    # Check for virtual environment flag
    if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
        # We're running in virtual environment, just run experiment
        run_experiment()
        print("=== Tesselation Experiment completed in virtual environment ===")
    else:
        # Run main setup function
        main()
