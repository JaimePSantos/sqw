#!/usr/bin/env python3

"""
Static noise experiment for quantum walks.

This script runs static noise experiments for quantum walks with configurable parameters.

Now uses smart loading from smart_loading_static module.

Execution Modes:
1. Full Pipeline (default): Compute samples + analysis + plots + archive
2. Samples Only: Set CALCULATE_SAMPLES_ONLY = True to only compute and save samples
3. Analysis Only: Set SKIP_SAMPLE_COMPUTATION = True to skip sample computation
4. Custom: Adjust individual toggles for plotting, archiving, etc.

This modular approach allows you to:
- Run expensive sample computation on cluster, then analysis locally
- Recompute analysis with different parameters without recomputing samples
- Split long computations into manageable chunks
"""

import time
import math
import numpy as np
import os
import sys
import subprocess
import signal
import tarfile
import traceback
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Import crash-safe logging decorator
from logging_module.crash_safe_logging import crash_safe_log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Plotting switch
ENABLE_PLOTTING = False  # Set to False to disable plotting
USE_LOGLOG_PLOT = False  # Set to True to use log-log scale for plotting
PLOT_FINAL_PROBDIST = False  # Set to True to plot probability distributions at final time step
SAVE_FIGURES = False  # Set to False to disable saving figures to files

# Archive switch
CREATE_TAR_ARCHIVE = True  # Set to True to create tar archive of experiments_data_samples folder

# Computation control switches
CALCULATE_SAMPLES_ONLY = True  # Set to True to only compute and save samples (skip analysis)
SKIP_SAMPLE_COMPUTATION = False  # Set to True to skip sample computation (analysis only)

# Check for environment variable overrides (from safe_background_launcher.py)
if os.environ.get('ENABLE_PLOTTING'):
    ENABLE_PLOTTING = os.environ.get('ENABLE_PLOTTING').lower() == 'true'
if os.environ.get('CREATE_TAR_ARCHIVE'):
    CREATE_TAR_ARCHIVE = os.environ.get('CREATE_TAR_ARCHIVE').lower() == 'true'
if os.environ.get('CALCULATE_SAMPLES_ONLY'):
    CALCULATE_SAMPLES_ONLY = os.environ.get('CALCULATE_SAMPLES_ONLY').lower() == 'true'
if os.environ.get('SKIP_SAMPLE_COMPUTATION'):
    SKIP_SAMPLE_COMPUTATION = os.environ.get('SKIP_SAMPLE_COMPUTATION').lower() == 'true'

# Background execution switch - SAFER IMPLEMENTATION
RUN_IN_BACKGROUND = True  # Set to True to automatically run the process in background

# Check if background execution has been disabled externally
if os.environ.get('RUN_IN_BACKGROUND') == 'False':
    RUN_IN_BACKGROUND = False
BACKGROUND_LOG_FILE = "static_experiment_background.log"  # Log file for background execution
BACKGROUND_PID_FILE = "static_experiment.pid"  # PID file to track background process

# Experiment parameters
N = 100  # System size (TEST: reduced for quick test)
steps = 10  # Time steps (TEST: reduced for quick test)
samples = 1  # Samples per deviation (TEST: reduced for quick test)

# Memory management warning
if N > 10000 and steps > 500:
    print(f"⚠️  WARNING: Large computation detected (N={N}, steps={steps})")
    print("   This may require significant memory and computation time.")
    print("   Consider reducing N or steps if you encounter memory issues.")
    print("   Each process may use several GB of RAM.")

# Check for forced parameter overrides from launcher
if os.environ.get('FORCE_SAMPLES_COUNT'):
    try:
        forced_samples = int(os.environ.get('FORCE_SAMPLES_COUNT'))
        print(f"🔒 FORCED: Using samples = {forced_samples} (launcher override)")
        samples = forced_samples
    except ValueError:
        pass

if os.environ.get('FORCE_N_VALUE'):
    try:
        forced_N = int(os.environ.get('FORCE_N_VALUE'))
        print(f"🔒 FORCED: Using N = {forced_N} (launcher override)")
        N = forced_N
        steps = N//4  # Recalculate steps
    except ValueError:
        pass

# Quantum walk parameters (for static noise, we only need theta)
theta = math.pi/3  # Base theta parameter for static noise
initial_state_kwargs = {"nodes": [N//2]}

# Deviation values for static noise experiments
# CUSTOMIZE THIS LIST: Add or remove deviation values as needed
devs = [0, 0.1]  # List of static noise deviation values (TEST: reduced for quick test)

# Multiprocessing configuration
# Reduce max processes to prevent memory exhaustion with large computations
MAX_PROCESSES = min(len(devs), max(1, mp.cpu_count() // 2))  # Use half available CPUs to prevent memory issues
PROCESS_LOG_DIR = "process_logs"  # Directory for individual process logs

# ============================================================================
# MULTIPROCESSING LOGGING SETUP
# ============================================================================

def setup_process_logging(dev_value, process_id):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_value:.3f}_pid_{process_id}.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_value}")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [DEV:%(name)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def setup_master_logging():
    """Setup logging for the master process"""
    master_log_filename = "static_experiment_multiprocess.log"
    
    # Create master logger
    master_logger = logging.getLogger("master")
    master_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in master_logger.handlers[:]:
        master_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(master_log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [MASTER] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    master_logger.addHandler(file_handler)
    master_logger.addHandler(console_handler)
    
    return master_logger, master_log_filename

# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================

def compute_dev_samples(dev_args):
    """Worker function to compute samples for a single deviation value in a separate process"""
    dev, process_id, N, steps, samples, theta, initial_state_kwargs = dev_args
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(dev, process_id)
    
    try:
        logger.info(f"Starting computation for deviation {dev:.4f}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.4f}")
        
        # Import required modules (each process needs its own imports)
        from sqw.experiments_expanded_static import running
        from smart_loading_static import get_experiment_dir
        import pickle
        import gc  # For garbage collection
        
        # Create dummy tessellation function
        def dummy_tesselation_func(N):
            return None
        
        # Setup experiment directory
        has_noise = dev > 0
        noise_params = [dev] if has_noise else [0]
        exp_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type="static_noise", 
                                   base_dir="experiments_data_samples")
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"Experiment directory: {exp_dir}")
        
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
                logger.info(f"Sample {sample_idx+1}/{samples} already exists, skipping")
                dev_computed_samples += 1
                continue
            
            logger.info(f"Computing sample {sample_idx+1}/{samples}...")
            
            # Extract initial nodes from initial_state_kwargs
            initial_nodes = initial_state_kwargs.get('nodes', [])
            
            # Memory-efficient approach: reduce steps if needed and process immediately
            try:
                # Run the quantum walk experiment for this sample using static noise
                logger.info(f"  Running quantum walk: N={N}, steps={steps}, dev={dev}")
                walk_result = running(
                    N, theta, steps,
                    initial_nodes=initial_nodes,
                    deviation_range=dev,
                    return_all_states=True
                )
                
                logger.info(f"  Computation completed, saving {len(walk_result)} steps...")
                
                # Save each step immediately and free memory
                for step_idx in range(len(walk_result)):
                    step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                    os.makedirs(step_dir, exist_ok=True)
                    
                    filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                    filepath = os.path.join(step_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        pickle.dump(walk_result[step_idx], f)
                    
                    # Progress logging for large computations
                    if step_idx % 100 == 0 or step_idx == len(walk_result) - 1:
                        logger.info(f"    Saved step {step_idx}/{len(walk_result)}")
                
                # Explicitly delete the result to free memory
                del walk_result
                logger.info(f"  All steps saved and memory freed")
                
            except MemoryError as mem_error:
                logger.error(f"Memory error during computation: {mem_error}")
                logger.error("Consider reducing N, steps, or running fewer processes simultaneously")
                raise
            except Exception as comp_error:
                logger.error(f"Computation error: {comp_error}")
                raise
            
            dev_computed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info(f"Sample {sample_idx+1}/{samples} completed in {sample_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        logger.info(f"Deviation {dev:.4f} completed: {dev_computed_samples} samples in {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": dev_computed_samples,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in process for dev {dev:.4f}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": 0,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

# ============================================================================
# STANDARD DEVIATION DATA MANAGEMENT
# ============================================================================

def create_or_load_std_data(mean_results, devs, N, steps, tesselation_func, std_base_dir, noise_type):
    """
    Create or load standard deviation data from mean probability distributions.
    
    Args:
        mean_results: List of mean probability distributions for each parameter
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        tesselation_func: Function to create tesselation (dummy for static noise)
        std_base_dir: Base directory for standard deviation data
        noise_type: Type of noise ("static_noise")
    
    Returns:
        List of standard deviation arrays for each deviation value
    """
    import os
    import pickle
    import numpy as np
    
    # Import functions from jaime_scripts and smart_loading_static
    from jaime_scripts import (
        prob_distributions2std
    )
    from smart_loading_static import get_experiment_dir
    
    print(f"\n[DATA] Managing standard deviation data in '{std_base_dir}'...")
    
    # Create base directory for std data
    os.makedirs(std_base_dir, exist_ok=True)
    
    stds = []
    domain = np.arange(N) - N//2  # Center domain around 0
    
    for i, dev in enumerate(devs):
        # Setup std data directory structure for static noise
        has_noise = dev > 0
        noise_params = [dev] if has_noise else [0]  # Static noise uses single parameter
        std_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type=noise_type, 
                                   base_dir=std_base_dir)
        os.makedirs(std_dir, exist_ok=True)
        
        std_filepath = os.path.join(std_dir, "std_vs_time.pkl")
        
        # Try to load existing std data
        if os.path.exists(std_filepath):
            try:
                with open(std_filepath, 'rb') as f:
                    std_values = pickle.load(f)
                print(f"  [OK] Loaded std data for dev {dev:.3f}")
                stds.append(std_values)
                continue
            except Exception as e:
                print(f"  [WARNING] Could not load std data for dev {dev:.3f}: {e}")
        
        # Compute std data from mean probability distributions
        print(f"  [COMPUTING] Computing std data for dev {dev:.3f}...")
        try:
            # Get mean probability distributions for this deviation
            if mean_results and i < len(mean_results) and mean_results[i]:
                dev_mean_prob_dists = mean_results[i]
                
                # Calculate standard deviations
                if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
                    std_values = prob_distributions2std(dev_mean_prob_dists, domain)
                    
                    # Save std data
                    with open(std_filepath, 'wb') as f:
                        pickle.dump(std_values, f)
                    
                    stds.append(std_values)
                    print(f"  [OK] Computed and saved std data for dev {dev:.3f} (final std = {std_values[-1]:.3f})")
                else:
                    print(f"  [ERROR] No valid probability distributions found for dev {dev:.3f}")
                    stds.append([])
            else:
                print(f"  [ERROR] No mean results available for dev {dev:.3f}")
                stds.append([])
                
        except Exception as e:
            print(f"  [ERROR] Error computing std data for dev {dev:.3f}: {e}")
            stds.append([])
    
    print(f"[OK] Standard deviation data management completed!")
    return stds

def create_experiment_archive(N, samples):
    """Create a tar archive of experiment data folders for the specific N value."""
    try:
        print("\n[ARCHIVE] Creating tar archive of experiment data...")
        
        # Create archive filename with timestamp, N, and samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"experiments_data_samples_N{N}_samples{samples}_{timestamp}.tar.gz"
        
        # Check if the experiments_data_samples folder exists
        data_folder = "experiments_data_samples"
        if not os.path.exists(data_folder):
            print(f"[WARNING] Data folder '{data_folder}' not found - skipping archive creation")
            return
        
        print(f"[ARCHIVE] Data folder found: {os.path.abspath(data_folder)}")
        
        n_folder_name = f"N_{N}"
        folders_to_archive = []
        
        # Find all folders containing N_{N} folders
        print(f"[ARCHIVE] Looking for folders containing '{n_folder_name}'...")
        
        for root, dirs, files in os.walk(data_folder):
            if n_folder_name in dirs:
                # Get the relative path from experiments_data_samples
                relative_root = os.path.relpath(root, data_folder)
                if relative_root == ".":
                    folder_path = n_folder_name
                else:
                    folder_path = os.path.join(relative_root, n_folder_name)
                
                full_path = os.path.join(data_folder, folder_path)
                folders_to_archive.append((full_path, folder_path))
                print(f"  Found: {folder_path}")
        
        if not folders_to_archive:
            print(f"[WARNING] No folders found containing '{n_folder_name}' - skipping archive creation")
            print(f"[DEBUG] Directory contents of {data_folder}:")
            try:
                for item in os.listdir(data_folder):
                    item_path = os.path.join(data_folder, item)
                    if os.path.isdir(item_path):
                        print(f"  Directory: {item}")
                        # List subdirectories to see if N_ folders exist deeper
                        try:
                            subdirs = [d for d in os.listdir(item_path) if os.path.isdir(os.path.join(item_path, d))]
                            if subdirs:
                                print(f"    Subdirs: {subdirs}")
                        except:
                            pass
                    else:
                        print(f"  File: {item}")
            except Exception as e:
                print(f"  Error listing directory: {e}")
            return
        
        print(f"[ARCHIVE] Creating archive: {archive_name}")
        
        # Create the tar archive with only N-specific folders
        with tarfile.open(archive_name, "w:gz") as tar:
            # Add the base experiments_data_samples structure but only with N-specific content
            for full_path, archive_path in folders_to_archive:
                print(f"  Adding to archive: {archive_path}")
                tar.add(full_path, arcname=os.path.join("experiments_data_samples", archive_path))
        
        # Get archive size
        archive_size = os.path.getsize(archive_name)
        size_mb = archive_size / (1024 * 1024)
        
        print(f"[OK] Archive created: {archive_name} ({size_mb:.1f} MB)")
        print(f"[OK] Archived {len(folders_to_archive)} folders containing N={N} data")
        print(f"[OK] Archive location: {os.path.abspath(archive_name)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to create archive: {e}")
        traceback.print_exc()

# @crash_safe_log(log_file_prefix="static_noise_experiment", heartbeat_interval=30.0)
def run_static_experiment():
    """Run the static noise quantum walk experiment with configurable execution modes."""
    
    # Validate configuration
    if CALCULATE_SAMPLES_ONLY and SKIP_SAMPLE_COMPUTATION:
        raise ValueError("Invalid configuration: Cannot set both CALCULATE_SAMPLES_ONLY=True and SKIP_SAMPLE_COMPUTATION=True")
    
    if SKIP_SAMPLE_COMPUTATION and not ENABLE_PLOTTING:
        print("WARNING: Analysis-only mode with plotting disabled - limited output will be generated")
    
    # Configuration summary
    print("=== EXECUTION CONFIGURATION ===")
    mode_description = (
        "Samples Only" if CALCULATE_SAMPLES_ONLY else
        "Analysis Only" if SKIP_SAMPLE_COMPUTATION else
        "Full Pipeline"
    )
    print(f"Execution mode: {mode_description}")
    print(f"Sample computation: {'Enabled' if not SKIP_SAMPLE_COMPUTATION else 'Disabled'}")
    print(f"Analysis phase: {'Enabled' if not CALCULATE_SAMPLES_ONLY else 'Disabled'}")
    print(f"Plotting: {'Enabled' if ENABLE_PLOTTING else 'Disabled'}")
    print(f"Archiving: {'Enabled' if CREATE_TAR_ARCHIVE else 'Disabled'}")
    print(f"Background execution: {'Enabled' if RUN_IN_BACKGROUND else 'Disabled'}")
    print("=" * 40)
    
    # Import required modules at the top
    import numpy as np
    import networkx as nx
    import pickle
    
    # SAFE Background execution handling
    if RUN_IN_BACKGROUND and not os.environ.get('IS_BACKGROUND_PROCESS'):
        print("Starting SAFE background execution...")
        
        try:
            script_path = os.path.abspath(__file__)
            # Use sys.executable to get the current Python interpreter path
            python_executable = sys.executable
            
            # Create environment for subprocess that prevents recursion
            env = os.environ.copy()
            env['IS_BACKGROUND_PROCESS'] = '1'  # This prevents infinite recursion
            
            # Create log and PID file paths
            log_file_path = os.path.join(os.getcwd(), BACKGROUND_LOG_FILE)
            pid_file_path = os.path.join(os.getcwd(), BACKGROUND_PID_FILE)
            
            # Check if there's already a background process running
            if os.path.exists(pid_file_path):
                try:
                    with open(pid_file_path, 'r') as f:
                        old_pid = int(f.read().strip())
                    
                    # Check if the old process is still running
                    if os.name == 'nt':  # Windows
                        result = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}"], 
                                              capture_output=True, text=True)
                        if str(old_pid) in result.stdout:
                            print(f"Background process already running (PID: {old_pid})")
                            print(f"   Kill it first with: taskkill /F /PID {old_pid}")
                            return
                    else:  # Unix-like
                        try:
                            os.kill(old_pid, 0)  # Check if process exists
                            print(f"Background process already running (PID: {old_pid})")
                            print(f"   Kill it first with: kill {old_pid}")
                            return
                        except OSError:
                            pass  # Process doesn't exist, continue
                            
                except (ValueError, OSError):
                    pass  # Invalid PID file, continue
            
            print("Starting background process...")
            
            # Initialize log file
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"Background execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Command: {python_executable} {script_path}\n")
                log_file.write("=" * 50 + "\n\n")
            
            if os.name == 'nt':  # Windows - SAFE METHOD
                # Use subprocess.Popen with proper flags to avoid process spam
                with open(log_file_path, 'a') as log_file:
                    process = subprocess.Popen(
                        [python_executable, "-u", script_path],  # -u for unbuffered output
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW  # Don't create visible window
                    )
                
                # Save PID for cleanup
                with open(pid_file_path, 'w') as pid_file:
                    pid_file.write(str(process.pid))
                
                # Give the process a moment to start and check if it's still running
                time.sleep(0.5)
                if process.poll() is None:
                    print(f"Background process started safely (PID: {process.pid})")
                else:
                    print(f"Warning: Background process (PID: {process.pid}) may have exited immediately")
                    print(f"Check log file for details: {log_file_path}")
                
            else:  # Unix-like systems - SAFE METHOD
                # Use nohup for proper background execution
                with open(log_file_path, 'a') as log_file:
                    # Try different approaches for Unix-like systems
                    process = None
                    
                    # First try with nohup and full detachment
                    try:
                        process = subprocess.Popen(
                            ["nohup", python_executable, "-u", script_path],
                            env=env,
                            cwd=os.getcwd(),
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid,  # Create new session
                            start_new_session=True  # Additional detachment on Python 3.7+
                        )
                    except (TypeError, AttributeError, OSError) as e:
                        print(f"   First attempt failed: {e}")
                        # Fallback 1: Try without start_new_session
                        try:
                            process = subprocess.Popen(
                                ["nohup", python_executable, "-u", script_path],
                                env=env,
                                cwd=os.getcwd(),
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid  # Create new session
                            )
                        except (AttributeError, OSError) as e2:
                            print(f"   Second attempt failed: {e2}")
                            # Fallback 2: Try without preexec_fn
                            try:
                                process = subprocess.Popen(
                                    ["nohup", python_executable, "-u", script_path],
                                    env=env,
                                    cwd=os.getcwd(),
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT
                                )
                            except OSError as e3:
                                print(f"   Third attempt failed: {e3}")
                                # Fallback 3: Try without nohup
                                process = subprocess.Popen(
                                    [python_executable, "-u", script_path],
                                    env=env,
                                    cwd=os.getcwd(),
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT
                                )
                
                # Save PID for cleanup
                with open(pid_file_path, 'w') as pid_file:
                    pid_file.write(str(process.pid))
                
                # Give the process a moment to start and check if it's still running
                time.sleep(0.5)
                if process.poll() is None:
                    print(f"Background process started safely (PID: {process.pid})")
                else:
                    print(f"Warning: Background process (PID: {process.pid}) may have exited immediately")
                    print(f"Check log file for details: {log_file_path}")
            
            print(f"Output logged to: {log_file_path}")
            print(f"Process ID saved to: {pid_file_path}")
            print("\n" + "="*50)
            print("SAFE BACKGROUND PROCESS STARTED")
            if os.name == 'nt':  # Windows
                print("   Monitor with: Get-Content " + BACKGROUND_LOG_FILE + " -Wait")
                print("   Kill with: taskkill /F /PID <pid>")
            else:  # Unix-like (Linux/macOS)
                print("   Monitor with: tail -f " + BACKGROUND_LOG_FILE)
                print("   Kill with: kill <pid>")
            print("="*50)
            
            return  # Exit the foreground process
            
        except Exception as e:
            print(f"Error starting background process: {e}")
            print("   Falling back to foreground execution...")
    
    # Check if we're the background process
    if os.environ.get('IS_BACKGROUND_PROCESS'):
        print("Running in SAFE background mode...")
        print(f"   Process ID: {os.getpid()}")
        print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Set up clean signal handlers for graceful shutdown
        def cleanup_and_exit(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            try:
                pid_file_path = os.path.join(os.getcwd(), BACKGROUND_PID_FILE)
                if os.path.exists(pid_file_path):
                    os.remove(pid_file_path)
                    print("Cleaned up PID file")
            except Exception as e:
                print(f"Warning during cleanup: {e}")
            print("Background process exiting cleanly")
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, cleanup_and_exit)
    
    # Import additional required modules
    try:
        # For static noise, we don't need tesselations module since it's built-in
        from sqw.experiments_expanded_static import running
        from sqw.states import uniform_initial_state, amp2prob
        
        # Import shared functions from jaime_scripts and smart_loading_static
        from jaime_scripts import (
            prob_distributions2std,
            plot_std_vs_time_qwak
        )
        from smart_loading_static import smart_load_or_create_experiment, get_experiment_dir
        
        print("Successfully imported all required modules")
    except ImportError as e:
        error_msg = f"Error: Could not import required modules: {e}"
        print(error_msg)
        print("Make sure you're running this script from the correct directory with all dependencies available")
        
        # If we're in background mode, write the error to the log file
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                with open(BACKGROUND_LOG_FILE, 'a') as f:
                    f.write(f"\n{error_msg}\n")
                    f.write("Script exiting due to import error\n")
            except:
                pass
        
        # Clean up PID file if we're the background process
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                if os.path.exists(BACKGROUND_PID_FILE):
                    os.remove(BACKGROUND_PID_FILE)
            except:
                pass
        
        raise

    print("Starting static noise quantum walk experiment...")
    
    # Memory safety check
    estimated_memory_per_process_mb = (steps * N * 16 * 2) / (1024 * 1024)  # 2x overhead
    total_estimated_memory_mb = estimated_memory_per_process_mb * MAX_PROCESSES
    
    print(f"Memory estimation:")
    print(f"  Per process: ~{estimated_memory_per_process_mb:.0f} MB")
    print(f"  Total (all processes): ~{total_estimated_memory_mb:.0f} MB")
    
    if total_estimated_memory_mb > 8000:  # 8GB threshold
        print("⚠️  WARNING: High memory usage predicted!")
        print("   Consider reducing N, steps, or MAX_PROCESSES")
        print("   Current settings may cause system instability")
    
    print(f"Experiment parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Multiprocessing: Using up to {MAX_PROCESSES} processes for {len(devs)} deviations")
    print("MULTIPROCESS MODE: Each deviation will run in a separate process!")
    
    # Setup master logging
    master_logger, master_log_file = setup_master_logging()
    master_logger.info("=" * 60)
    master_logger.info("MULTIPROCESS STATIC NOISE EXPERIMENT STARTED")
    master_logger.info("=" * 60)
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    master_logger.info(f"Master log file: {master_log_file}")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0

    # Create a dummy tessellation function for static noise
    def dummy_tesselation_func(N):
        """Dummy tessellation function for static noise (tessellations are built-in)"""
        return None

    # Sample computation phase with multiprocessing
    experiment_time = 0
    process_results = []
    
    if not SKIP_SAMPLE_COMPUTATION:
        master_logger.info("=" * 40)
        master_logger.info("MULTIPROCESS SAMPLE COMPUTATION PHASE")
        master_logger.info("=" * 40)
        
        # Prepare arguments for each process
        process_args = []
        for process_id, dev in enumerate(devs):
            args = (dev, process_id, N, steps, samples, theta, initial_state_kwargs)
            process_args.append(args)
        
        master_logger.info(f"Launching {len(process_args)} processes...")
        
        # Track process information
        process_info = {}
        for i, (dev, process_id, *_) in enumerate(process_args):
            log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev:.3f}_pid_{process_id}.log")
            process_info[dev] = {
                "process_id": process_id,
                "log_file": log_file,
                "start_time": None,
                "end_time": None,
                "status": "pending"
            }
        
        # Execute processes concurrently
        try:
            with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
                # Submit all jobs
                future_to_dev = {}
                for args in process_args:
                    dev = args[0]
                    future = executor.submit(compute_dev_samples, args)
                    future_to_dev[future] = dev
                    process_info[dev]["start_time"] = time.time()
                    process_info[dev]["status"] = "running"
                    master_logger.info(f"Process launched for dev={dev:.4f} (PID will be assigned)")
                
                # Collect results as they complete
                for future in as_completed(future_to_dev):
                    dev = future_to_dev[future]
                    try:
                        result = future.result()
                        process_results.append(result)
                        process_info[dev]["end_time"] = time.time()
                        
                        if result["success"]:
                            process_info[dev]["status"] = "completed"
                            completed_samples += result["computed_samples"]
                            master_logger.info(f"✓ Process for dev={dev:.4f} completed successfully")
                            master_logger.info(f"  Computed samples: {result['computed_samples']}")
                            master_logger.info(f"  Time: {result['total_time']:.1f}s")
                            master_logger.info(f"  Log file: {result['log_file']}")
                        else:
                            process_info[dev]["status"] = "failed"
                            master_logger.error(f"✗ Process for dev={dev:.4f} failed")
                            master_logger.error(f"  Error: {result['error']}")
                            master_logger.error(f"  Log file: {result['log_file']}")
                            
                    except Exception as e:
                        import traceback
                        process_info[dev]["status"] = "failed"
                        process_info[dev]["end_time"] = time.time()
                        error_msg = f"Exception in process for dev={dev:.4f}: {str(e)}"
                        master_logger.error(error_msg)
                        master_logger.error(traceback.format_exc())
                        
                        process_results.append({
                            "dev": dev,
                            "process_id": -1,
                            "computed_samples": 0,
                            "total_time": 0,
                            "log_file": "unknown",
                            "success": False,
                            "error": error_msg
                        })
        
        except Exception as e:
            import traceback
            master_logger.error(f"Critical error in multiprocessing: {str(e)}")
            master_logger.error(traceback.format_exc())
            raise

        experiment_time = time.time() - start_time
        
        # Log final results
        master_logger.info("=" * 40)
        master_logger.info("MULTIPROCESS COMPUTATION COMPLETED")
        master_logger.info("=" * 40)
        master_logger.info(f"Total execution time: {experiment_time:.2f} seconds")
        master_logger.info(f"Total samples computed: {completed_samples}/{total_samples}")
        
        # Log individual process results
        master_logger.info("\nPROCESS SUMMARY:")
        successful_processes = 0
        failed_processes = 0
        
        for result in process_results:
            dev = result["dev"]
            if result["success"]:
                successful_processes += 1
                master_logger.info(f"  ✓ dev={dev:.4f}: {result['computed_samples']} samples in {result['total_time']:.1f}s")
            else:
                failed_processes += 1
                master_logger.error(f"  ✗ dev={dev:.4f}: FAILED - {result['error']}")
        
        master_logger.info(f"\nRESULTS: {successful_processes} successful, {failed_processes} failed processes")
        
        # Log process log file locations
        master_logger.info("\nPROCESS LOG FILES:")
        for dev, info in process_info.items():
            master_logger.info(f"  dev={dev:.4f}: {info['log_file']}")
        
        print(f"\n[COMPLETED] Multiprocess sample computation completed in {experiment_time:.2f} seconds")
        print(f"Total samples computed: {completed_samples}")
        print(f"Successful processes: {successful_processes}/{len(devs)}")
        print(f"Master log file: {master_log_file}")
        print(f"Process log directory: {PROCESS_LOG_DIR}")
        
    else:
        master_logger.info("SKIPPING SAMPLE COMPUTATION")
        print("=== SKIPPING SAMPLE COMPUTATION ===")
        print("Sample computation disabled - proceeding to analysis phase")
        experiment_time = 0
        completed_samples = 0

    # Early exit if only computing samples
    if CALCULATE_SAMPLES_ONLY:
        master_logger.info("SAMPLES ONLY MODE - ANALYSIS SKIPPED")
        print("\n=== SAMPLES ONLY MODE - ANALYSIS SKIPPED ===")
        print("Sample computation completed. Skipping analysis and plotting.")
        
        # Create tar archive if enabled (even in samples-only mode)
        if CREATE_TAR_ARCHIVE:
            master_logger.info("Creating tar archive...")
            create_experiment_archive(N, samples)
        else:
            master_logger.info("Archiving disabled")
            print("Archiving disabled (CREATE_TAR_ARCHIVE=False)")
        
        print("To run analysis on existing samples, set:")
        print("  CALCULATE_SAMPLES_ONLY = False")
        print("  SKIP_SAMPLE_COMPUTATION = True")
        
        total_time = time.time() - start_time
        master_logger.info(f"Total execution time: {total_time:.2f} seconds")
        master_logger.info("Experiment completed (samples only mode)")
        
        print(f"Total execution time: {total_time:.2f} seconds")
        
        return {
            "mode": "samples_only",
            "devs": devs,
            "N": N,
            "steps": steps,
            "samples": samples,
            "total_time": total_time,
            "theta": theta,
            "completed_samples": completed_samples,
            "multiprocessing": True,
            "process_results": process_results,
            "master_log_file": master_log_file,
            "process_log_dir": PROCESS_LOG_DIR
        }

    # Analysis phase
    print("\n=== ANALYSIS PHASE ===")
    print("Loading existing samples and computing analysis...")

    # Smart load or create mean probability distributions
    print("\n[DATA] Smart loading/creating mean probability distributions...")
    try:
        mean_results = smart_load_or_create_experiment(
            graph_func=lambda n: None,  # Not used in static noise
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            angles_or_angles_list=theta,  # Single theta value for static noise
            tesselation_order_or_list=None,  # Not used in static noise
            initial_state_func=uniform_initial_state,
            initial_state_kwargs=initial_state_kwargs,
            parameter_list=devs,
            samples=samples,
            noise_type="static_noise",
            parameter_name="static_dev",
            samples_base_dir="experiments_data_samples",
            probdist_base_dir="experiments_data_samples_probDist"
        )
        print("[OK] Mean probability distributions ready for analysis")
        
    except Exception as e:
        print(f"[WARNING] Warning: Could not smart load/create mean probability distributions: {e}")
        mean_results = None

    # Create or load standard deviation data
    try:
        stds = create_or_load_std_data(
            mean_results, devs, N, steps, dummy_tesselation_func,
            "experiments_data_samples_std", "static_noise"
        )
        
        # Print final std values for verification
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values and len(std_values) > 0:
                print(f"Dev {dev:.3f}: Final std = {std_values[-1]:.3f}")
            else:
                print(f"Dev {dev:.3f}: No valid standard deviation data")
                
    except Exception as e:
        print(f"[WARNING] Warning: Could not create/load standard deviation data: {e}")
        stds = []

    # Plot standard deviation vs time if enabled
    if ENABLE_PLOTTING:
        print("\n[PLOT] Creating standard deviation vs time plot...")
        try:
            if 'stds' in locals() and len(stds) > 0 and any(len(std) > 0 for std in stds):
                import matplotlib.pyplot as plt
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                for i, (std_values, dev) in enumerate(zip(stds, devs)):
                    if len(std_values) > 0:
                        time_steps = list(range(len(std_values)))
                        
                        # Filter out zero values for log-log plot
                        if USE_LOGLOG_PLOT:
                            # Remove zero values which can't be plotted on log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                            if filtered_data:
                                filtered_times, filtered_stds = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=f'Static deviation = {dev:.3f}', 
                                         marker='o', markersize=3, linewidth=2)


                        else:
                            plt.plot(time_steps, std_values, 
                                   label=f'Static deviation = {dev:.3f}', 
                                   marker='o', markersize=3, linewidth=2)
                
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                
                if USE_LOGLOG_PLOT:
                    plt.title('Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3, which="both", ls="-")  # Grid for both major and minor ticks
                    plot_filename = "static_noise_std_vs_time_loglog.png"
                else:
                    plt.title('Standard Deviation vs Time for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plot_filename = "static_noise_std_vs_time.png"
                
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # Save the plot (if enabled)
                if SAVE_FIGURES:
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"[OK] Plot saved as '{plot_filename}'")
                
                # Show the plot
                plt.show()
                plot_type = "log-log" if USE_LOGLOG_PLOT else "linear"
                saved_status = " and saved" if SAVE_FIGURES else ""
                print(f"[OK] Standard deviation plot displayed{saved_status}! (Scale: {plot_type})")
            else:
                print("[WARNING] Warning: No standard deviation data available for plotting")
        except Exception as e:
            print(f"[WARNING] Warning: Could not create plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[PLOT] Plotting disabled (ENABLE_PLOTTING=False)")

    # Plot final probability distributions if enabled
    if ENABLE_PLOTTING and PLOT_FINAL_PROBDIST:
        print("\n[PLOT] Creating final probability distribution plot...")
        try:
            if 'mean_results' in locals() and mean_results and len(mean_results) > 0:
                import matplotlib.pyplot as plt
                
                # Create the plot
                plt.figure(figsize=(14, 8))
                
                # Use the last time step (steps-1)
                final_step = steps - 1
                domain = np.arange(N) - N//2  # Center domain around 0
                
                for i, (dev_mean_prob_dists, dev) in enumerate(zip(mean_results, devs)):
                    if dev_mean_prob_dists and len(dev_mean_prob_dists) > final_step and dev_mean_prob_dists[final_step] is not None:
                        final_prob_dist = dev_mean_prob_dists[final_step].flatten()
                        
                        # Plot the probability distribution with log y-axis
                        plt.semilogy(domain, final_prob_dist, 
                                   label=f'Static deviation = {dev:.3f}', 
                                   linewidth=2, alpha=0.8)
                
                plt.xlabel('Position', fontsize=12)
                plt.ylabel('Probability (log scale)', fontsize=12)
                plt.title(f'Probability Distributions at Final Time Step (t={final_step}) - Log Scale', fontsize=14)
                plt.xlim(-150, 150)  # Limit x-axis range to -150 to 150
                plt.ylim(1e-20, None)  # Limit y-axis minimum to 10^-20
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot (if enabled)
                probdist_filename = "static_noise_final_probdist_log.png"
                if SAVE_FIGURES:
                    plt.savefig(probdist_filename, dpi=300, bbox_inches='tight')
                    print(f"[OK] Probability distribution plot saved as '{probdist_filename}'")
                
                # Show the plot
                plt.show()
                saved_status = " and saved" if SAVE_FIGURES else ""
                print(f"[OK] Final probability distribution plot displayed{saved_status}!")
            else:
                print("[WARNING] Warning: No mean probability distribution data available for plotting")
        except Exception as e:
            print(f"[WARNING] Warning: Could not create probability distribution plot: {e}")
            import traceback
            traceback.print_exc()
    elif not ENABLE_PLOTTING:
        print("\n[PLOT] Probability distribution plotting disabled (ENABLE_PLOTTING=False)")
    else:
        print("\n[PLOT] Final probability distribution plotting disabled (PLOT_FINAL_PROBDIST=False)")

    # Create tar archive if enabled
    if CREATE_TAR_ARCHIVE:
        create_experiment_archive(N, samples)

    print("Static noise experiment completed successfully!")
    total_time = time.time() - start_time
    master_logger.info(f"EXPERIMENT COMPLETED - Total time: {total_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n=== Performance Summary ===")
    print(f"Execution mode: {'Samples Only' if CALCULATE_SAMPLES_ONLY else 'Analysis Only' if SKIP_SAMPLE_COMPUTATION else 'Full Pipeline'}")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(devs)}")
    print(f"Multiprocessing: {MAX_PROCESSES} max processes")
    
    if not SKIP_SAMPLE_COMPUTATION:
        print(f"Total quantum walks computed: {completed_samples}")
        successful_processes = len([r for r in process_results if r["success"]])
        print(f"Successful processes: {successful_processes}/{len(devs)}")
        if experiment_time > 0 and completed_samples > 0:
            print(f"Average time per quantum walk: {experiment_time / completed_samples:.3f} seconds")
    else:
        print(f"Expected quantum walks: {len(devs) * samples} (sample computation skipped)")
    
    print("\n=== Multiprocessing Log Files ===")
    print(f"Master log: {master_log_file}")
    print(f"Process logs directory: {PROCESS_LOG_DIR}")
    if process_results:
        print("Individual process logs:")
        for result in process_results:
            status = "✓" if result["success"] else "✗"
            print(f"  {status} dev={result['dev']:.4f}: {result['log_file']}")
    
    print("\n=== Execution Modes ===")
    print("Available execution modes:")
    print("1. Full Pipeline (default): Compute samples + analysis + plots + archive")
    print("2. Samples Only: Set CALCULATE_SAMPLES_ONLY = True")
    print("3. Analysis Only: Set SKIP_SAMPLE_COMPUTATION = True")
    print("4. Custom: Adjust individual toggles for plotting, archiving, etc.")
    
    print("\n=== Static Noise Details ===")
    print("Static noise model:")
    print(f"- dev=0: Perfect static evolution with theta={theta:.3f} (no noise)")
    print("- dev>0: Random deviation applied to Hamiltonian edges with range 'dev'")
    print("- Each sample generates different random noise for edge parameters")
    print("- Mean probability distributions average over all samples")
    print("- Tessellations are built-in (alpha and beta patterns)")
    print("- MULTIPROCESSING: Each deviation value runs in separate process")
    
    print("\n=== Plotting Features ===")
    print(f"- Plotting enabled: {ENABLE_PLOTTING}")
    print(f"- Save figures to files: {SAVE_FIGURES}")
    if ENABLE_PLOTTING:
        plot_type = "Log-log scale" if USE_LOGLOG_PLOT else "Linear scale"
        plot_filename = "static_noise_std_vs_time_loglog.png" if USE_LOGLOG_PLOT else "static_noise_std_vs_time.png"
        print(f"- Standard deviation plot type: {plot_type}")
        if SAVE_FIGURES:
            print(f"- Standard deviation plot saved as: {plot_filename}")
        if USE_LOGLOG_PLOT:
            print("- Log-log plots help identify power-law scaling behavior sigma(t) proportional to t^alpha")
        
        print(f"- Final probability distribution plot enabled: {PLOT_FINAL_PROBDIST}")
        if PLOT_FINAL_PROBDIST:
            if SAVE_FIGURES:
                print("- Final probability distribution plot saved as: static_noise_final_probdist_log.png")
            print("- Shows probability distributions at the final time step for all deviations")
            print("- Uses log scale for y-axis and focuses on position range -150 to +150")
    
    print("\n=== Archive Features ===")
    print(f"- Create tar archive: {CREATE_TAR_ARCHIVE}")
    if CREATE_TAR_ARCHIVE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"experiments_data_samples_N{N}_samples{samples}_{timestamp}.tar.gz"
        print(f"- Archive will be saved as: experiments_data_samples_N{N}_samples{samples}_[timestamp].tar.gz")
        print(f"- Archive contains only N={N} folders and their parent directory structure")
        print("- This selective archiving reduces file size compared to archiving all N values")
    
    return {
        "mode": "full_pipeline",
        "devs": devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time,
        "theta": theta,
        "completed_samples": completed_samples,
        "sample_computation_enabled": not SKIP_SAMPLE_COMPUTATION,
        "analysis_enabled": not CALCULATE_SAMPLES_ONLY,
        "plotting_enabled": ENABLE_PLOTTING,
        "archiving_enabled": CREATE_TAR_ARCHIVE,
        "multiprocessing": True,
        "max_processes": MAX_PROCESSES,
        "process_results": process_results,
        "master_log_file": master_log_file,
        "process_log_dir": PROCESS_LOG_DIR
    }

if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    try:
        run_static_experiment()
    except Exception as e:
        error_msg = f"Fatal error in run_static_experiment: {e}"
        print(error_msg)
        
        # Try to log to master logger if available
        try:
            master_logger = logging.getLogger("master")
            if master_logger.handlers:
                master_logger.error(error_msg)
                master_logger.error(traceback.format_exc())
        except:
            pass
        
        # If we're in background mode, write the error to the log file
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                with open(BACKGROUND_LOG_FILE, 'a') as f:
                    f.write(f"\n{error_msg}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.write("\nScript exiting due to fatal error\n")
            except:
                pass
            
            # Clean up PID file if we're the background process
            try:
                if os.path.exists(BACKGROUND_PID_FILE):
                    os.remove(BACKGROUND_PID_FILE)
                    print("Cleaned up PID file due to error")
            except:
                pass
        
        # Re-raise the exception if not in background mode
        if not os.environ.get('IS_BACKGROUND_PROCESS'):
            raise
        else:
            import sys
            sys.exit(1)