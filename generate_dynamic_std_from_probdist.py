#!/usr/bin/env python3

"""
Generate Standard Deviation Data from Dynamic Probability Distributions

This script generates standard deviation data from existing dynamic probability distribution files.
It processes multiple deviation values in parallel, calculating standard deviations across
time for each deviation and saving the results for later plotting.

Key Features:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if std files exist and have valid data)
- Dynamic noise experiment directory structure handling
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_dynamic_std_from_probdist.py

Configuration:
    Edit the parameters section below to match your experiment setup.
"""

import os
import gc
import time
import math
import signal
import logging
import traceback
import multiprocessing as mp
import pickle
import tarfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime

# Try to import psutil for system monitoring
try:
    import psutil
except ImportError:
    psutil = None

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create per-dev and main tar archives
ARCHIVE_DIR = "experiments_archive_dynamic"

# # Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
# N = 20000              # System size (production scale for cluster)
# steps = N//4           # Time steps (5000 for N=20000)
# samples = 40           # Samples per deviation (full production count)
# base_theta = math.pi/3 # Base theta parameter for dynamic angle noise

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 100              # System size (production scale for cluster)
steps = N//4           # Time steps (5000 for N=20000)
samples = 1          # Samples per deviation (full production count)
base_theta = math.pi/3 # Base theta parameter for dynamic angle noise

# Deviation values - Dynamic noise format (angle deviations) - Matching original static experiment
devs = [
    0,                  # No noise (equivalent to (0,0))
    0.2,                # Small noise (equivalent to (0, 0.2))
    0.6,                # Medium noise (equivalent to (0, 0.6))
    0.8,                # Medium noise (equivalent to (0, 0.8))
    1.0,                # Large noise (equivalent to (0, 1))
]

# Directory configuration
PROBDIST_BASE_DIR = "experiments_data_samples_dynamic_probDist"
STD_BASE_DIR = "experiments_data_samples_dynamic_std"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_dynamic_std")

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())

# Timeout configuration - Scale with problem size
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {MAX_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_dynamic_std", "dynamic_std_generation_master.log")

# Global shutdown flag
SHUTDOWN_REQUESTED = False

# ============================================================================
# SIGNAL HANDLING AND UTILITIES
# ============================================================================

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SHUTDOWN] Received signal {signum}. Initiating graceful shutdown...")
    print("[SHUTDOWN] Waiting for current processes to complete...")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def dummy_tesselation_func(N):
    """Dummy tessellation function for dynamic noise (tessellations are built-in)"""
    return None

# ============================================================================
# SYSTEM MONITORING AND LOGGING UTILITIES
# ============================================================================

def log_system_resources(logger=None, prefix="[SYSTEM]"):
    """Log current system resource usage"""
    try:
        if psutil:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            msg = f"{prefix} Memory: {memory.percent:.1f}% used ({memory.available / (1024**3):.1f}GB free), CPU: {cpu_percent:.1f}%"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            
            # Check for concerning resource usage
            if memory.percent > 90:
                logger.warning(f"{prefix} High memory usage: {memory.percent:.1f}%") if logger else print(f"WARNING: {prefix} High memory usage: {memory.percent:.1f}%")
            
            if cpu_percent > 95:
                logger.warning(f"{prefix} High CPU usage: {cpu_percent:.1f}%") if logger else print(f"WARNING: {prefix} High CPU usage: {cpu_percent:.1f}%")
                
    except ImportError:
        msg = f"{prefix} psutil not available - cannot monitor resources"
        if logger:
            logger.info(msg)
        else:
            print(msg)
    except Exception as e:
        msg = f"{prefix} Error monitoring resources: {e}"
        if logger:
            logger.error(msg)
        else:
            print(msg)

def log_progress_update(phase, completed, total, start_time, logger=None):
    """Log detailed progress update with ETA"""
    elapsed = time.time() - start_time
    if completed > 0:
        eta = (elapsed / completed) * (total - completed)
        eta_str = f"{eta/60:.1f}m" if eta < 3600 else f"{eta/3600:.1f}h"
        progress_pct = (completed / total) * 100
        
        msg = f"[{phase}] Progress: {completed}/{total} ({progress_pct:.1f}%) - Elapsed: {elapsed/60:.1f}m - ETA: {eta_str}"
    else:
        msg = f"[{phase}] Progress: {completed}/{total} (0.0%) - Elapsed: {elapsed/60:.1f}m - ETA: unknown"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_process_logging(dev_value, process_id, base_theta=None):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format dev_value for filename with rounding to 6 decimal places
    dev_rounded = round(float(dev_value), 6)
    dev_str = f"{dev_rounded:.6f}"
    
    # Format base_theta for filename
    if base_theta is not None:
        theta_str = f"_basetheta{base_theta:.6f}"
    else:
        theta_str = ""
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}{theta_str}_dynamic_std.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_dynamic_std")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [DEV:%(name)s] %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('[DEV:%(name)s] %(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def setup_master_logging():
    """Setup logging for the master process"""
    # Create master logger
    master_logger = logging.getLogger("master")
    master_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in master_logger.handlers[:]:
        master_logger.removeHandler(handler)
    
    # Ensure the directory exists for master log file
    master_log_dir = os.path.dirname(MASTER_LOG_FILE)
    if master_log_dir:
        os.makedirs(master_log_dir, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(MASTER_LOG_FILE, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter('[%(asctime)s] [MASTER] %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    master_logger.addHandler(file_handler)
    master_logger.addHandler(console_handler)
    
    return master_logger

# ============================================================================
# DYNAMIC EXPERIMENT DIRECTORY FUNCTIONS
# ============================================================================

def get_dynamic_experiment_dir(tesselation_func, has_noise, N, noise_params=None, base_dir="experiments_data_samples_dynamic", base_theta=None):
    """
    Get experiment directory for dynamic noise experiments.
    
    Structure: base_dir/tesselation_name/[noise|no_noise]/basetheta_X/dev_X/N_X/
    """
    if tesselation_func is None or tesselation_func.__name__ == "dummy_tesselation_func":
        tesselation_name = "dynamic_angle_noise"
    else:
        tesselation_name = tesselation_func.__name__
    
    # Create the main experiment directory structure
    exp_base = os.path.join(base_dir, tesselation_name)
    
    # Add noise subdirectory
    if has_noise:
        noise_dir = os.path.join(exp_base, "noise")
    else:
        noise_dir = os.path.join(exp_base, "no_noise")
    
    # Add base theta information if provided
    if base_theta is not None:
        theta_str = f"basetheta_{base_theta:.6f}".replace(".", "p")
        theta_dir = os.path.join(noise_dir, theta_str)
    else:
        theta_dir = noise_dir
    
    # Add deviation-specific directory
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        dev_rounded = round(dev, 6)
        dev_str = f"dev_{dev_rounded:.6f}".replace(".", "p")
        dev_dir = os.path.join(theta_dir, dev_str)
    else:
        dev_dir = theta_dir
    
    # Add N-specific directory
    final_dir = os.path.join(dev_dir, f"N_{N}")
    
    return final_dir

def find_dynamic_probdist_directory_for_config(base_dir, N, base_theta, dev, logger):
    """
    Find the dynamic probability distribution directory that contains data for the specified configuration.
    
    Returns:
        tuple: (probdist_dir_path, format_type) or (None, None) if not found
    """
    logger.info(f"Searching for dynamic probdist directory: N={N}, dev={dev}, base_theta={base_theta:.6f}")
    
    # Dynamic format structure
    has_noise = dev > 0
    noise_params = [dev]
    
    probdist_path = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=base_dir, 
        base_theta=base_theta
    )
    
    if os.path.exists(probdist_path):
        # Check if we have probability distribution files
        probdist_files = [f for f in os.listdir(probdist_path) if f.startswith("mean_step_") and f.endswith(".pkl")]
        found_steps = len(probdist_files)
        
        logger.info(f"Found dynamic probdist directory: {probdist_path}")
        logger.info(f"  Steps: {found_steps}")
        
        return probdist_path, "dynamic_format"
    
    logger.warning(f"No valid dynamic probdist directory found for configuration: N={N}, dev={dev}")
    return None, None

# ============================================================================
# DIRECTORY AND FILE MANAGEMENT
# ============================================================================

def validate_std_file(file_path):
    """
    Validate that a standard deviation file exists and contains valid data.
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a valid array with expected properties
        if isinstance(data, (list, np.ndarray)) and len(data) > 0:
            # Additional validation: check for reasonable std values (allow 0, but not negative)
            if hasattr(data, '__iter__') and all(x >= 0 for x in data if x is not None):
                return True
        
        return False
        
    except (pickle.PickleError, EOFError, ValueError, TypeError) as e:
        return False

def load_probability_distributions_for_dev(probdist_dir, N, steps, logger):
    """
    Load all probability distributions for a single deviation from probDist files.
    
    Args:
        probdist_dir: Directory containing probability distribution files
        N: System size
        steps: Number of time steps
        logger: Logger for this process
    
    Returns:
        list: List of probability distributions for each time step
    """
    try:
        logger.info(f"Loading probability distributions from: {probdist_dir}")
        
        prob_distributions = []
        
        # Get available probdist files
        probdist_files = [f for f in os.listdir(probdist_dir) if f.startswith("mean_step_") and f.endswith(".pkl")]
        available_steps = []
        for f in probdist_files:
            try:
                step_num = int(f.replace("mean_step_", "").replace(".pkl", ""))
                available_steps.append(step_num)
            except ValueError:
                continue
        
        available_steps.sort()
        max_step = max(available_steps) if available_steps else 0
        actual_steps = max_step + 1  # Steps are 0-indexed
        
        logger.info(f"Found {len(available_steps)} probdist files, max step: {max_step}")
        
        for step_idx in range(actual_steps):
            # Log every 100th step or first/last step
            if step_idx % 100 == 0 or step_idx == actual_steps - 1:
                logger.info(f"Loading step {step_idx}/{actual_steps - 1} (probability distribution for time step {step_idx})")
            
            prob_file = os.path.join(probdist_dir, f"mean_step_{step_idx}.pkl")
            
            if os.path.exists(prob_file):
                try:
                    with open(prob_file, 'rb') as f:
                        prob_dist = pickle.load(f)
                    prob_distributions.append(prob_dist)
                except Exception as e:
                    logger.warning(f"Failed to load probability distribution for step {step_idx}: {e}")
                    prob_distributions.append(None)
            else:
                logger.warning(f"Probability distribution file not found for step {step_idx}: {prob_file}")
                prob_distributions.append(None)
            
            # Progress summary every 100 steps or at the end
            if step_idx % 100 == 0 or step_idx == actual_steps - 1:
                valid_steps_so_far = sum(1 for p in prob_distributions if p is not None)
                logger.info(f"Progress: {step_idx + 1}/{actual_steps} steps loaded ({valid_steps_so_far} valid distributions)")
        
        valid_steps = sum(1 for p in prob_distributions if p is not None)
        logger.info(f"Loaded {valid_steps}/{len(prob_distributions)} valid probability distributions")
        
        return prob_distributions
        
    except Exception as e:
        logger.error(f"Error loading probability distributions: {str(e)}")
        return []

# ============================================================================
# STANDARD DEVIATION CALCULATION FUNCTIONS
# ============================================================================

def prob_distributions2std(prob_distributions, domain):
    """
    Calculate standard deviations from probability distributions.
    
    Args:
        prob_distributions: List of probability distributions for each time step
        domain: Domain array (positions)
    
    Returns:
        List of standard deviation values for each time step
    """
    std_values = []
    
    for prob_dist in prob_distributions:
        if prob_dist is None:
            std_values.append(0)
            continue
            
        try:
            # Ensure probability distribution is properly formatted
            prob_dist_flat = prob_dist.flatten()
            total_prob = np.sum(prob_dist_flat)
            
            if total_prob == 0:
                std_values.append(0)
                continue
                
            # Always normalize to ensure proper probability distribution
            prob_dist_flat = prob_dist_flat / total_prob
            
            # Calculate 1st moment (mean position)
            moment_1 = np.sum(domain * prob_dist_flat)
            
            # Calculate 2nd moment
            moment_2 = np.sum(domain**2 * prob_dist_flat)
            
            # Calculate standard deviation: sqrt(moment(2) - moment(1)^2)
            stDev = moment_2 - moment_1**2
            std = np.sqrt(stDev) if stDev > 0 else 0
            std_values.append(std)
            
        except Exception as e:
            std_values.append(0)
    
    return std_values

def generate_dynamic_std_for_dev(dev_args):
    """
    Worker function to generate standard deviation data for a single deviation value.
    
    Args:
        dev_args: Tuple containing (dev, process_id, N, steps, samples, base_theta)
    
    Returns:
        dict: Results from the standard deviation generation process
    """
    dev, process_id, N, steps, samples_count, base_theta_param = dev_args
    
    # Setup logging for this process
    dev_rounded = round(dev, 6)
    dev_str = f"{dev_rounded:.6f}"
    logger, log_file = setup_process_logging(dev_str, process_id, base_theta_param)
    
    try:
        logger.info(f"=== DYNAMIC STANDARD DEVIATION GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, base_theta={base_theta_param:.6f}")
        
        dev_start_time = time.time()
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        
        # Handle deviation format for has_noise check
        has_noise = dev > 0
        
        # Get source and target directories
        noise_params = [dev]
        probdist_exp_dir, probdist_format = find_dynamic_probdist_directory_for_config(
            PROBDIST_BASE_DIR, N, base_theta_param, dev, logger
        )
        
        std_exp_dir = get_dynamic_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, 
            base_dir=STD_BASE_DIR, 
            base_theta=base_theta_param
        )
        
        logger.info(f"ProbDist source: {probdist_exp_dir}")
        logger.info(f"Std target: {std_exp_dir}")
        logger.info(f"Source format: {probdist_format}")
        
        # Check if probDist directory exists
        if probdist_exp_dir is None or not os.path.exists(probdist_exp_dir):
            logger.error(f"ProbDist directory not found: {probdist_exp_dir}")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": f"ProbDist directory not found: {probdist_exp_dir}",
                "log_file": log_file, "total_time": 0, "dev_tar_path": None
            }
        
        # Check if std file already exists and is valid
        os.makedirs(std_exp_dir, exist_ok=True)
        std_file = os.path.join(std_exp_dir, "std_vs_time.pkl")
        
        skipped = False
        if validate_std_file(std_file):
            logger.info(f"Valid std file already exists: {std_file}")
            skipped = True
        
        valid_std_count = 0
        std_values = []
        if not skipped:
            # Load probability distributions
            prob_distributions = load_probability_distributions_for_dev(probdist_exp_dir, N, steps, logger)
            if not prob_distributions or len(prob_distributions) == 0:
                logger.error(f"No probability distributions loaded")
                return {
                    "dev": dev, "process_id": process_id, "success": False,
                    "error": "No probability distributions loaded",
                    "log_file": log_file, "total_time": 0, "dev_tar_path": None
                }
            
            # Calculate standard deviations
            logger.info(f"Calculating standard deviations for {len(prob_distributions)} time steps...")
            domain = np.arange(N) - N//2  # Center domain around 0
            std_values = prob_distributions2std(prob_distributions, domain)
            valid_std_count = sum(1 for s in std_values if s is not None and s > 0)
            
            logger.info(f"Calculated {valid_std_count}/{len(std_values)} valid standard deviations")
            
            if valid_std_count == 0:
                logger.error(f"No valid standard deviations calculated")
                return {
                    "dev": dev, "process_id": process_id, "success": False,
                    "error": "No valid standard deviations calculated",
                    "log_file": log_file, "total_time": 0, "dev_tar_path": None
                }
            
            # Save standard deviation data
            with open(std_file, 'wb') as f:
                pickle.dump(std_values, f)
            logger.info(f"Standard deviation data saved to: {std_file}")
            
            if valid_std_count > 0:
                final_std = [s for s in std_values if s is not None and s > 0][-1]
                logger.info(f"Final std value: {final_std:.4f}")

        else:
            # If skipped, load the std_values for reporting
            try:
                with open(std_file, 'rb') as f:
                    std_values = pickle.load(f)
                valid_std_count = sum(1 for s in std_values if s is not None and s > 0)
            except Exception:
                std_values = []
                valid_std_count = 0

        dev_time = time.time() - dev_start_time
        logger.info(f"=== DYNAMIC STANDARD DEVIATION GENERATION COMPLETED ===")
        logger.info(f"Valid std values: {valid_std_count}/{len(std_values)}")
        logger.info(f"Total time: {dev_time:.1f}s")

        # Archive this dev's std directory if requested (always, even if skipped)
        dev_tar_path = None
        if CREATE_TAR:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            # Format dev string for filename
            devstr = f"{dev_rounded:.6f}".replace(".", "p")
            # Create archive of the std directory
            dev_tar_path = os.path.join(ARCHIVE_DIR, f"std_dynamic_basetheta{base_theta_param:.6f}_dev_{devstr}_N{N}.tar")
            if os.path.exists(std_exp_dir):
                with tarfile.open(dev_tar_path, "w") as tar:
                    tar.add(std_exp_dir, arcname=f"dynamic_std_dev_{devstr}_N{N}")
                logger.info(f"Created temporary archive: {dev_tar_path}")
            else:
                logger.warning(f"Std directory does not exist for archiving: {std_exp_dir}")

        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "error": None,
            "valid_std_count": valid_std_count,
            "total_std_count": len(std_values),
            "log_file": log_file,
            "total_time": dev_time,
            "action": "skipped" if skipped else "generated",
            "message": "Valid std file already exists" if skipped else f"Generated {valid_std_count} valid std values",
            "dev_tar_path": dev_tar_path
        }
        
    except Exception as e:
        dev_rounded = round(dev, 6)
        dev_str = f"{dev_rounded:.4f}"
        error_msg = f"Error in dynamic std generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": False,
            "error": error_msg,
            "log_file": log_file,
            "total_time": 0,
            "dev_tar_path": None
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for dynamic standard deviation generation."""
    
    print("=== DYNAMIC STANDARD DEVIATION GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"ProbDist source: {PROBDIST_BASE_DIR}")
    print(f"Std target: {STD_BASE_DIR}")
    if CREATE_TAR:
        print(f"Archiving: {ARCHIVE_DIR}")
    print("=" * 60)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== DYNAMIC STANDARD DEVIATION GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, base_theta)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for dynamic std generation...")
    
    process_results = []
    
    # Track process information
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        process_info[dev] = {"index": i, "process_id": process_id, "status": "queued"}
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_dynamic_std_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
                
                # Update process status
                process_info[args[0]]["status"] = "running"
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    # Update process status
                    process_info[dev]["status"] = "completed"
                    
                    if result["success"]:
                        dev_rounded = round(dev, 6)
                        action = result.get("action", "processed")
                        message = result.get("message", "")
                        master_logger.info(f"Dev {dev_rounded:.4f}: SUCCESS - "
                                         f"{action.upper()} - {message}, "
                                         f"Time: {result['total_time']:.1f}s")
                    else:
                        dev_rounded = round(dev, 6)
                        master_logger.error(f"Dev {dev_rounded:.4f}: FAILED - {result['error']}")
                        
                except TimeoutError:
                    dev_rounded = round(dev, 6)
                    error_msg = f"Dev {dev_rounded:.4f}: TIMEOUT after {PROCESS_TIMEOUT}s"
                    master_logger.error(error_msg)
                    process_results.append({
                        "dev": dev, "success": False, "error": error_msg,
                        "valid_std_count": 0, "total_std_count": 0, "total_time": 0,
                        "dev_tar_path": None
                    })
                    process_info[dev]["status"] = "timeout"
                except Exception as e:
                    dev_rounded = round(dev, 6)
                    error_msg = f"Dev {dev_rounded:.4f}: EXCEPTION - {str(e)}"
                    master_logger.error(error_msg)
                    process_results.append({
                        "dev": dev, "success": False, "error": error_msg,
                        "valid_std_count": 0, "total_std_count": 0, "total_time": 0,
                        "dev_tar_path": None
                    })
                    process_info[dev]["status"] = "error"
    
    except KeyboardInterrupt:
        master_logger.warning("Interrupted by user")
        print("\n[INTERRUPT] Gracefully shutting down processes...")
    except Exception as e:
        master_logger.error(f"Critical error in multiprocessing: {str(e)}")
        raise

    # === MAIN ARCHIVE CREATION ===
    if CREATE_TAR:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_tar_name = f"experiments_data_dynamic_std_N{N}_samples{samples}_basetheta{base_theta:.6f}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        dev_tar_paths = [r.get("dev_tar_path") for r in process_results if r.get("dev_tar_path")]
        if dev_tar_paths:
            master_logger.info(f"Creating archive: {main_tar_path}")
            with tarfile.open(main_tar_path, "w") as tar:
                for dev_tar in dev_tar_paths:
                    tar.add(dev_tar, arcname=os.path.basename(dev_tar))
            print(f"Created main archive: {main_tar_path}")
            master_logger.info(f"Created main archive: {main_tar_path}")
            # Delete temporary dev tar files
            for dev_tar in dev_tar_paths:
                try:
                    os.remove(dev_tar)
                    master_logger.info(f"Deleted temporary archive: {dev_tar}")
                except Exception as e:
                    master_logger.warning(f"Could not delete {dev_tar}: {e}")
        else:
            master_logger.info("No archives created by processes")
    
    total_time = time.time() - start_time
    
    # Generate summary
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    generated_count = sum(1 for r in process_results if r.get("action") == "generated")
    skipped_count = sum(1 for r in process_results if r.get("action") == "skipped")
    
    print(f"\n=== DYNAMIC STD GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Actions: {generated_count} generated, {skipped_count} skipped")
    if CREATE_TAR:
        print(f"Archives: {ARCHIVE_DIR}/")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== DYNAMIC STD GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Actions: {generated_count} generated, {skipped_count} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            action = result.get("action", "processed")
            message = result.get("message", "")
            master_logger.info(f"  Dev {result['dev']}: SUCCESS - "
                             f"{action.upper()} - {message}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Dev {result['dev']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "generated_count": generated_count,
        "skipped_count": skipped_count,
        "process_results": process_results
    }

if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        raise
