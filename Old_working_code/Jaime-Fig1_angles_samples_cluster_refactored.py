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
        "matplotlib",
        "qwak-sim"
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
    """Run the actual quantum walk experiment with cluster-optimized immediate saving."""
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
    
    print(f"Cluster-optimized parameters: N={N}, steps={steps}, samples={samples}")
    print(f"This will significantly reduce computation time while preserving the experiment structure.")
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
            
            # Run the walk - note the correct parameter order
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
    
    # Now create mean probability distributions
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
    
    # Load the mean results for statistics
    try:
        mean_results = load_mean_probability_distributions(
            even_line_two_tesselation, N, steps, devs, "experiments_data_samples_probDist", "angle"
        )
        print("✅ Mean probability distributions loaded for analysis")
    except Exception as e:
        print(f"⚠️  Warning: Could not load mean probability distributions: {e}")
        mean_results = [[] for _ in devs]

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

    print("Angle experiment completed successfully!")
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
    
    print("\n=== Angle Noise Details ===")
    print("Angle noise model:")
    print("- dev=0: Perfect angles [π/3, π/3] (no noise)")
    print("- dev>0: Random deviation from [π/3, π/3] with standard deviation 'dev'")
    print("- Each sample generates a different random angle sequence")
    print("- Mean probability distributions average over all samples")
    
    print("\n=== Scaling Note ===")
    print("For production runs, you can scale up the parameters:")
    print("- Increase N for higher resolution")
    print("- Increase steps for longer evolution")
    print("- Increase samples for better statistics")
    print("- Add more deviation values as needed")
    print("- Adjust deviations to explore different noise regimes")
    
    # Create TAR archive of results
    print("Creating TAR archive of results...")
    archive_filename = zip_results("experiments_data_samples", "experiments_data_samples_probDist", N, samples)
    
    print("=== Analysis Instructions ===")
    if archive_filename:
        print(f"Results archived in: {archive_filename}")
    print("To analyze the results, transfer the tar file and extract it, then use:")
    print("- experiments_data_samples/ contains the raw quantum states for each sample")
    print("- experiments_data_samples_probDist/ contains the mean probability distributions")
    print("Both directories maintain the angle directory structure for easy analysis.")
    print("\nDirectory structure:")
    print("experiments_data_samples/even_line_two_tesselation_angle_nonoise/N_1000/")
    print("experiments_data_samples/even_line_two_tesselation_angle_noise/angle_dev_X.XXX/N_1000/")

if __name__ == "__main__":
    # Check for virtual environment flag
    if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
        # We're running in virtual environment, just run experiment
        run_experiment()
        print("=== Experiment completed in virtual environment ===")
    else:
        # Run main setup function
        main()
