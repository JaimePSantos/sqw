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

# Import crash-safe logging decorator
from logging_module.crash_safe_logging import crash_safe_log

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Plotting switch
ENABLE_PLOTTING = True  # Set to False to disable plotting
USE_LOGLOG_PLOT = True  # Set to True to use log-log scale for plotting
PLOT_FINAL_PROBDIST = True  # Set to True to plot probability distributions at final time step
SAVE_FIGURES = True  # Set to False to disable saving figures to files

# Archive switches
CREATE_TAR_ARCHIVE = True  # Set to True to create tar archive
ARCHIVE_SAMPLES = True     # Set to True to archive experiments_data_samples folder
ARCHIVE_PROBDIST = True    # Set to True to archive experiments_data_samples_probDist folder

# Computation control switches - EXPLICIT CONTROL
CALCULATE_SAMPLES_ONLY = True  # Set to True to only compute and save samples (skip analysis)
SKIP_SAMPLE_COMPUTATION = False  # Set to True to skip sample computation (analysis only)

# Detailed computation control (when not in samples-only or analysis-only modes)
COMPUTE_RAW_SAMPLES = True      # Compute quantum walk samples 
COMPUTE_PROBDIST = True         # Compute probability distributions from samples
COMPUTE_STD_DATA = False         # Compute standard deviation data from probability distributions

# Check for environment variable overrides (from safe_background_launcher.py)
if os.environ.get('ENABLE_PLOTTING'):
    ENABLE_PLOTTING = os.environ.get('ENABLE_PLOTTING').lower() == 'true'
if os.environ.get('CREATE_TAR_ARCHIVE'):
    CREATE_TAR_ARCHIVE = os.environ.get('CREATE_TAR_ARCHIVE').lower() == 'true'
if os.environ.get('ARCHIVE_SAMPLES'):
    ARCHIVE_SAMPLES = os.environ.get('ARCHIVE_SAMPLES').lower() == 'true'
if os.environ.get('ARCHIVE_PROBDIST'):
    ARCHIVE_PROBDIST = os.environ.get('ARCHIVE_PROBDIST').lower() == 'true'
if os.environ.get('CALCULATE_SAMPLES_ONLY'):
    CALCULATE_SAMPLES_ONLY = os.environ.get('CALCULATE_SAMPLES_ONLY').lower() == 'true'
if os.environ.get('SKIP_SAMPLE_COMPUTATION'):
    SKIP_SAMPLE_COMPUTATION = os.environ.get('SKIP_SAMPLE_COMPUTATION').lower() == 'true'
if os.environ.get('COMPUTE_RAW_SAMPLES'):
    COMPUTE_RAW_SAMPLES = os.environ.get('COMPUTE_RAW_SAMPLES').lower() == 'true'
if os.environ.get('COMPUTE_PROBDIST'):
    COMPUTE_PROBDIST = os.environ.get('COMPUTE_PROBDIST').lower() == 'true'
if os.environ.get('COMPUTE_STD_DATA'):
    COMPUTE_STD_DATA = os.environ.get('COMPUTE_STD_DATA').lower() == 'true'

# Background execution switch - SAFER IMPLEMENTATION
RUN_IN_BACKGROUND = False  # Set to True to automatically run the process in background

# Check if background execution has been disabled externally
if os.environ.get('RUN_IN_BACKGROUND') == 'False':
    RUN_IN_BACKGROUND = False
BACKGROUND_LOG_FILE = "static_experiment_background.log"  # Log file for background execution
BACKGROUND_PID_FILE = "static_experiment.pid"  # PID file to track background process

# Experiment parameters
N = 20000  # System size
steps = N//4  # Time steps
samples = 5  # Samples per deviation - changed from 1 to 5

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

# List of static noise deviations
devs = [0, 0.1, 0.5,1, 10]

# Note: Set USE_LOGLOG_PLOT = True in the plotting configuration above to use log-log scale
# This is useful for identifying power-law behavior in the standard deviation growth

# ============================================================================
# STANDARD DEVIATION DATA MANAGEMENT
# ============================================================================

def create_or_load_std_data(mean_results, devs, N, steps, tesselation_func, std_base_dir, noise_type, theta=None):
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
                                   base_dir=std_base_dir, theta=theta)
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
        
        # Check what we're archiving
        archive_items = []
        if ARCHIVE_SAMPLES:
            archive_items.append("samples")
        if ARCHIVE_PROBDIST:
            archive_items.append("probdist")
        
        if not archive_items:
            print("[WARNING] No archive items selected (ARCHIVE_SAMPLES=False, ARCHIVE_PROBDIST=False)")
            return
        
        print(f"[ARCHIVE] Archive contents: {', '.join(archive_items)}")
        
        # Create archive filename with timestamp, N, samples, and content type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_suffix = "_".join(archive_items)
        archive_name = f"experiments_data_{content_suffix}_N{N}_samples{samples}_{timestamp}.tar.gz"
        
        # Data directories to potentially archive
        data_directories = []
        if ARCHIVE_SAMPLES:
            data_directories.append(("experiments_data_samples", "samples"))
        if ARCHIVE_PROBDIST:
            data_directories.append(("experiments_data_samples_probDist", "probability distributions"))
        
        all_folders_to_archive = []
        n_folder_name = f"N_{N}"
        
        # Process each data directory
        for data_folder, folder_type in data_directories:
            if not os.path.exists(data_folder):
                print(f"[WARNING] {folder_type.capitalize()} folder '{data_folder}' not found - skipping")
                continue
            
            print(f"[ARCHIVE] Processing {folder_type} folder: {os.path.abspath(data_folder)}")
            
            # Find all folders containing N_{N} folders
            print(f"[ARCHIVE] Looking for folders containing '{n_folder_name}' in {folder_type}...")
            
            for root, dirs, files in os.walk(data_folder):
                if n_folder_name in dirs:
                    # Get the relative path from data folder
                    relative_root = os.path.relpath(root, data_folder)
                    if relative_root == ".":
                        folder_path = n_folder_name
                    else:
                        folder_path = os.path.join(relative_root, n_folder_name)
                    
                    full_path = os.path.join(data_folder, folder_path)
                    archive_path = os.path.join(data_folder, folder_path)
                    all_folders_to_archive.append((full_path, archive_path, folder_type))
                    print(f"  Found: {folder_path} ({folder_type})")
        
        if not all_folders_to_archive:
            print(f"[WARNING] No folders found containing '{n_folder_name}' in any data directories")
            
            # Debug information for each directory
            for data_folder, folder_type in data_directories:
                if os.path.exists(data_folder):
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
            # Add folders from each data directory
            for full_path, archive_path, folder_type in all_folders_to_archive:
                print(f"  Adding to archive: {archive_path} ({folder_type})")
                tar.add(full_path, arcname=archive_path)
        
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
    
    # Override detailed switches when using high-level modes
    if CALCULATE_SAMPLES_ONLY:
        COMPUTE_RAW_SAMPLES = True
        COMPUTE_PROBDIST = False
        COMPUTE_STD_DATA = False
        print("SAMPLES ONLY mode: Forcing COMPUTE_RAW_SAMPLES=True, COMPUTE_PROBDIST=False, COMPUTE_STD_DATA=False")
    
    if SKIP_SAMPLE_COMPUTATION:
        COMPUTE_RAW_SAMPLES = False
        # Keep COMPUTE_PROBDIST and COMPUTE_STD_DATA as they were set (allows selective analysis)
        print("ANALYSIS ONLY mode: Forcing COMPUTE_RAW_SAMPLES=False")
    
    if CREATE_TAR_ARCHIVE and not ARCHIVE_SAMPLES and not ARCHIVE_PROBDIST:
        print("WARNING: Archive enabled but no content selected (ARCHIVE_SAMPLES=False, ARCHIVE_PROBDIST=False)")
        print("         Archive creation will be skipped")
    
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
    
    # High-level computation control
    print(f"Sample computation: {'Enabled' if not SKIP_SAMPLE_COMPUTATION else 'Disabled'}")
    print(f"Analysis phase: {'Enabled' if not CALCULATE_SAMPLES_ONLY else 'Disabled'}")
    
    # Detailed computation control (when in full pipeline mode)
    if not CALCULATE_SAMPLES_ONLY and not SKIP_SAMPLE_COMPUTATION:
        print(f"  - Raw samples computation: {'Enabled' if COMPUTE_RAW_SAMPLES else 'Disabled'}")
        print(f"  - Probability dist computation: {'Enabled' if COMPUTE_PROBDIST else 'Disabled'}")
        print(f"  - Standard deviation computation: {'Enabled' if COMPUTE_STD_DATA else 'Disabled'}")
    
    print(f"Plotting: {'Enabled' if ENABLE_PLOTTING else 'Disabled'}")
    
    # Archive configuration
    if CREATE_TAR_ARCHIVE:
        archive_items = []
        if ARCHIVE_SAMPLES:
            archive_items.append("samples")
        if ARCHIVE_PROBDIST:
            archive_items.append("probdist")
        if archive_items:
            print(f"Archiving: Enabled ({', '.join(archive_items)})")
        else:
            print("Archiving: Enabled but no content selected")
    else:
        print("Archiving: Disabled")
    
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
    
    print(f"Experiment parameters: N={N}, steps={steps}, samples={samples}")
    print("IMMEDIATE SAVE MODE: Each sample will be saved as soon as it's computed!")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0

    # Create a dummy tessellation function for static noise
    def dummy_tesselation_func(N):
        """Dummy tessellation function for static noise (tessellations are built-in)"""
        return None

    # Sample computation phase
    experiment_time = 0
    completed_samples = 0
    
    if not SKIP_SAMPLE_COMPUTATION and COMPUTE_RAW_SAMPLES:
        print("=== SAMPLE COMPUTATION PHASE ===")
        print("Computing and saving quantum walk samples...")
        
        # Run experiments with immediate saving for each sample
        for dev_idx, dev in enumerate(devs):
            print(f"\n=== Processing static noise deviation {dev:.4f} ({dev_idx+1}/{len(devs)}) ===")
            
            # Setup experiment directory
            has_noise = dev > 0
            noise_params = [dev] if has_noise else [0]  # Static noise uses single parameter
            exp_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, noise_params=noise_params, noise_type="static_noise", base_dir="experiments_data_samples", theta=theta)
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
                    print(f"  [OK] Sample {sample_idx+1}/{samples} already exists, skipping")
                    completed_samples += 1
                    continue
                
                print(f"  [COMPUTING] Computing sample {sample_idx+1}/{samples}...")
                
                # For static noise, we don't need to generate angle sequences
                # The noise is applied internally by the running function
                deviation_range = dev
                
                # Extract initial nodes from initial_state_kwargs
                initial_nodes = initial_state_kwargs.get('nodes', [])
                
                # Run the quantum walk experiment for this sample using static noise
                walk_result = running(
                    N, theta, steps,
                    initial_nodes=initial_nodes,
                    deviation_range=deviation_range,
                    return_all_states=True
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
                
                print(f"  [OK] Sample {sample_idx+1}/{samples} saved in {sample_time:.1f}s")
                print(f"     Progress: {completed_samples}/{total_samples} ({progress_pct:.1f}%)")
                print(f"     Elapsed: {elapsed_time:.1f}s, Remaining: ~{remaining_time:.1f}s")
            
            dev_time = time.time() - dev_start_time
            print(f"[OK] Static noise deviation {dev:.4f} completed: {dev_computed_samples} new samples in {dev_time:.1f}s")

        experiment_time = time.time() - start_time
        print(f"\n[COMPLETED] Sample computation completed in {experiment_time:.2f} seconds")
        print(f"Total samples computed: {completed_samples}")
    elif SKIP_SAMPLE_COMPUTATION:
        print("=== SKIPPING SAMPLE COMPUTATION ===")
        print("Sample computation disabled - proceeding to analysis phase")
        experiment_time = 0
        completed_samples = 0
    else:
        print("=== SKIPPING RAW SAMPLE COMPUTATION ===") 
        print("Raw sample computation disabled (COMPUTE_RAW_SAMPLES=False) - proceeding to analysis phase")
        experiment_time = 0
        completed_samples = 0

    # Early exit if only computing samples
    if CALCULATE_SAMPLES_ONLY:
        print("\n=== SAMPLES ONLY MODE - ANALYSIS SKIPPED ===")
        print("Sample computation completed. Skipping analysis and plotting.")
        
        # Create tar archive if enabled (even in samples-only mode)
        if CREATE_TAR_ARCHIVE:
            create_experiment_archive(N, samples)
        else:
            print("Archiving disabled (CREATE_TAR_ARCHIVE=False)")
        
        print("To run analysis on existing samples, set:")
        print("  CALCULATE_SAMPLES_ONLY = False")
        print("  SKIP_SAMPLE_COMPUTATION = True")
        
        total_time = time.time() - start_time
        print(f"Total execution time: {total_time:.2f} seconds")
        
        return {
            "mode": "samples_only",
            "devs": devs,
            "N": N,
            "steps": steps,
            "samples": samples,
            "total_time": total_time,
            "theta": theta,
            "completed_samples": completed_samples
        }

    # Analysis phase
    print("\n=== ANALYSIS PHASE ===")
    print("Loading existing samples and computing analysis...")

    # Smart load or create mean probability distributions
    mean_results = None
    if COMPUTE_PROBDIST:
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
                probdist_base_dir="experiments_data_samples_probDist",
                theta=theta
            )
            print("[OK] Mean probability distributions ready for analysis")
            
        except Exception as e:
            print(f"[WARNING] Warning: Could not smart load/create mean probability distributions: {e}")
            mean_results = None
    else:
        print("\n[SKIPPED] Probability distribution computation disabled (COMPUTE_PROBDIST=False)")
        mean_results = None

    # Create or load standard deviation data
    stds = []
    if COMPUTE_STD_DATA:
        try:
            stds = create_or_load_std_data(
                mean_results, devs, N, steps, dummy_tesselation_func,
                "experiments_data_samples_std", "static_noise", theta=theta
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
    else:
        print("\n[SKIPPED] Standard deviation computation disabled (COMPUTE_STD_DATA=False)")
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
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n=== Performance Summary ===")
    print(f"Execution mode: {'Samples Only' if CALCULATE_SAMPLES_ONLY else 'Analysis Only' if SKIP_SAMPLE_COMPUTATION else 'Full Pipeline'}")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(devs)}")
    
    if not SKIP_SAMPLE_COMPUTATION:
        print(f"Total quantum walks computed: {completed_samples}")
        if experiment_time > 0 and completed_samples > 0:
            print(f"Average time per quantum walk: {experiment_time / completed_samples:.3f} seconds")
    else:
        print(f"Expected quantum walks: {len(devs) * samples} (sample computation skipped)")
    
    print("\n=== Execution Modes ===")
    print("Available execution modes:")
    print("1. Full Pipeline (default): Compute samples + analysis + plots + archive")
    print("2. Samples Only: Set CALCULATE_SAMPLES_ONLY = True")
    print("3. Analysis Only: Set SKIP_SAMPLE_COMPUTATION = True")
    print("4. Custom: Adjust individual toggles for plotting, archiving, etc.")
    print()
    print("Detailed computation control (Full Pipeline mode):")
    print("- COMPUTE_RAW_SAMPLES: Generate quantum walk samples")
    print("- COMPUTE_PROBDIST: Create probability distributions from samples")
    print("- COMPUTE_STD_DATA: Calculate standard deviation data from prob dists")
    print("Note: High-level modes (CALCULATE_SAMPLES_ONLY/SKIP_SAMPLE_COMPUTATION)")
    print("      automatically override detailed switches for consistency")
    
    print("\n=== Static Noise Details ===")
    print("Static noise model:")
    print(f"- dev=0: Perfect static evolution with theta={theta:.3f} (no noise)")
    print("- dev>0: Random deviation applied to Hamiltonian edges with range 'dev'")
    print("- Each sample generates different random noise for edge parameters")
    print("- Mean probability distributions average over all samples")
    print("- Tessellations are built-in (alpha and beta patterns)")
    
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
        print(f"- Archive samples data: {ARCHIVE_SAMPLES}")
        print(f"- Archive probability distributions: {ARCHIVE_PROBDIST}")
        
        # Determine what's being archived
        archive_items = []
        if ARCHIVE_SAMPLES:
            archive_items.append("samples")
        if ARCHIVE_PROBDIST:
            archive_items.append("probdist")
        
        if archive_items:
            content_suffix = "_".join(archive_items)
            archive_name = f"experiments_data_{content_suffix}_N{N}_samples{samples}_[timestamp].tar.gz"
            print(f"- Archive will be saved as: {archive_name}")
            print(f"- Archive contains only N={N} folders and their parent directory structure")
            print("- This selective archiving reduces file size compared to archiving all N values")
            
            if ARCHIVE_SAMPLES:
                print("- Samples data: Raw quantum walk samples from experiments_data_samples/")
            if ARCHIVE_PROBDIST:
                print("- Probability distributions: Mean probability distributions from experiments_data_samples_probDist/")
        else:
            print("- Warning: Archive enabled but no content selected (ARCHIVE_SAMPLES=False, ARCHIVE_PROBDIST=False)")
    
    print("\n=== Archive Content Control ===")
    print("Archive content can be controlled with these switches:")
    print("- ARCHIVE_SAMPLES: Include raw quantum walk samples")
    print("- ARCHIVE_PROBDIST: Include computed probability distributions")
    print("- Both can be True for complete archive, or set individually for specific content")
    
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
        "archiving_enabled": CREATE_TAR_ARCHIVE
    }

if __name__ == "__main__":
    try:
        run_static_experiment()
    except Exception as e:
        error_msg = f"Fatal error in run_static_experiment: {e}"
        print(error_msg)
        
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