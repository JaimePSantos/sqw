#!/usr/bin/env python3

"""
Generate Standard Deviation Data from Probability Distributions

This script generates standard deviation data from existing probability distribution files.
It processes multiple deviation values in parallel, calculdef get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data_samples", theta=None, samples=None):
    """
    Get experiment directory path with proper structure.
    
    Structure: base_dir/tesselation_func_noise_type/theta_value/dev_range/N_value/samples_count
    Example: experiments_data_samples/dummy_tesselation_func_static_noise/theta_1.047198/dev_min0.000_max0.000/N_300/samples_5
    """tandard deviations across
time for each deviation and saving the results for later plotting.

Key Features:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if std files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_std_from_probdist.py

Configuration:
    Edit the parameters section below to match your experiment setup.
"""

import os
import sys
import time
import math
import signal
import pickle
import logging
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import traceback
import tarfile

# ============================================================================
# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create per-dev and main tar archives
ARCHIVE_DIR = "experiments_archive"

# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 20000                # System size
steps = N//4           # Time steps
samples = 40           # Samples per deviation
theta = math.pi/3      # Theta parameter for static noise

# Deviation values - TEST SET (matching other scripts)
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),           # Medium noise range  
]


# Directory configuration
PROBDIST_BASE_DIR = "experiments_data_samples_probDist"
STD_BASE_DIR = "experiments_data_samples_std"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_std")

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())

# Timeout configuration - Scale with problem size (matches static_cluster_logged_mp.py)
# Base timeout for each process, scaled by N and steps
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {MAX_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_std", "std_generation_master.log")

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
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

# ============================================================================
# SYSTEM MONITORING AND LOGGING UTILITIES
# ============================================================================

def log_system_resources(logger=None, prefix="[SYSTEM]"):
    """Log current system resource usage (matches static_cluster_logged_mp.py)"""
    try:
        import psutil
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
    """Log detailed progress update with ETA (matches static_cluster_logged_mp.py)"""
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

def setup_process_logging(dev_value, process_id):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format dev_value for filename
    if isinstance(dev_value, (tuple, list)) and len(dev_value) == 2:
        min_val, max_val = dev_value
        dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        dev_str = f"{float(dev_value):.3f}"
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_std.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_std")
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
# DIRECTORY AND FILE MANAGEMENT
# ============================================================================

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data_samples", theta=None, samples=None):
    """
    Get experiment directory path with proper structure.
    
    Structure: base_dir/tesselation_func_noise_type/theta_value/dev_range/N_value/samples_count
    Example: experiments_data_samples_probDist/dummy_tesselation_func_static_noise/theta_1.047198/dev_min0.000_max0.000/N_20000/samples_40
    """
    # Handle deviation format
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            dev_str = f"dev_min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"dev_{float(dev):.3f}"
    else:
        dev_str = "dev_min0.000_max0.000"
    
    # Format theta for directory
    if theta is not None:
        theta_str = f"theta_{theta:.6f}"
    else:
        theta_str = "theta_default"
    
    # Format tessellation function name for directory
    if tesselation_func is None or hasattr(tesselation_func, '__name__') and tesselation_func.__name__ == 'dummy_tesselation_func':
        tessellation_name = "dummy_tesselation_func"
    else:
        tessellation_name = getattr(tesselation_func, '__name__', 'unknown_tesselation_func')
    
    # Build directory path with correct structure
    exp_dir = os.path.join(
        base_dir,
        f"{tessellation_name}_{noise_type}",
        theta_str,
        dev_str,
        f"N_{N}"
    )
    
    if samples is not None:
        exp_dir = os.path.join(exp_dir, f"samples_{samples}")
    
    return exp_dir

def find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data_samples", theta=None):
    """
    Find experiment directory with flexible format matching.
    Tries different directory structures to find existing data.
    """
    # Try new format first (with and without samples in path)
    for samples_try in [40, 20, 10, 5, None]:  # Common sample counts
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta, samples_try)
        if os.path.exists(exp_dir):
            return exp_dir, "new_format"
    
    # Try old format variations for backwards compatibility
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"{float(dev):.3f}"
    else:
        dev_str = "0.000"
    
    # Try various old format possibilities
    old_formats = [
        os.path.join(base_dir, f"N_{N}", f"static_dev_{dev_str}"),
        os.path.join(base_dir, f"N_{N}", f"dev_{dev_str}"),
        os.path.join(base_dir, f"static_dev_{dev_str}", f"N_{N}"),
    ]
    
    for exp_dir in old_formats:
        if os.path.exists(exp_dir):
            return exp_dir, "old_format"
    
    # If nothing found, return the new format path (will be created if needed)
    return get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta, None), "new_format"

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
        
        for step_idx in range(steps):
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
        
        valid_steps = sum(1 for p in prob_distributions if p is not None)
        logger.info(f"Loaded {valid_steps}/{steps} valid probability distributions")
        
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

def generate_std_for_dev(dev_args):
    """
    Worker function to generate standard deviation data for a single deviation value.
    
    Args:
        dev_args: Tuple containing (dev, process_id, N, steps, samples, theta)
    
    Returns:
        dict: Results from the standard deviation generation process
    """
    dev, process_id, N, steps, samples_count, theta_param = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"=== STANDARD DEVIATION GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        
        dev_start_time = time.time()
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        
        # Handle deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            has_noise = max_val > 0
        else:
            has_noise = dev > 0
        
        # Get source and target directories
        noise_params = [dev]
        probdist_exp_dir, probdist_format = find_experiment_dir_flexible(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=PROBDIST_BASE_DIR, theta=theta_param
        )
        
        std_exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=STD_BASE_DIR, theta=theta_param, samples=samples_count
        )
        
        logger.info(f"ProbDist source: {probdist_exp_dir}")
        logger.info(f"Std target: {std_exp_dir}")
        logger.info(f"Source format: {probdist_format}")
        
        # Check if probDist directory exists
        if not os.path.exists(probdist_exp_dir):
            logger.error(f"ProbDist directory not found: {probdist_exp_dir}")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": f"ProbDist directory not found: {probdist_exp_dir}",
                "log_file": log_file, "total_time": 0
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
                    "log_file": log_file, "total_time": 0
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
                    "log_file": log_file, "total_time": 0
                }
            # Save standard deviation data
            with open(std_file, 'wb') as f:
                pickle.dump(std_values, f)
            logger.info(f"Standard deviation data saved to: {std_file}")
            if valid_std_count > 0:
                final_std = [s for s in std_values if s is not None and s > 0][-1]
                logger.info(f"Final std value: {final_std:.6f}")
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
        logger.info(f"=== STANDARD DEVIATION GENERATION COMPLETED ===")
        logger.info(f"Valid std values: {valid_std_count}/{len(std_values)}")
        logger.info(f"Total time: {dev_time:.1f}s")

        # Archive this dev's std directory if requested (always, even if skipped)
        dev_tar_path = None
        if CREATE_TAR:
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            # Format dev string for filename
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                min_val, max_val = dev
                devstr = f"min{min_val:.3f}_max{max_val:.3f}"
            else:
                devstr = f"{float(dev):.3f}"
            # Find the theta_ folder containing this std_exp_dir
            theta_dir = os.path.dirname(os.path.dirname(std_exp_dir))
            theta_folder_name = os.path.basename(theta_dir)
            dev_folder_name = os.path.basename(os.path.dirname(std_exp_dir))
            dev_dir = os.path.dirname(std_exp_dir)
            dev_tar_path = os.path.join(ARCHIVE_DIR, f"std_{theta_folder_name}_{devstr}.tar")
            with tarfile.open(dev_tar_path, "w") as tar:
                tar.add(dev_dir, arcname=os.path.join(theta_folder_name, dev_folder_name))
            logger.info(f"Created temporary archive: {dev_tar_path}")

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
        error_msg = f"Error in std generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": False,
            "error": error_msg,
            "log_file": log_file,
            "total_time": 0
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for standard deviation generation."""
    
    print("=== STANDARD DEVIATION GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"ProbDist source: {PROBDIST_BASE_DIR}")
    print(f"Std target: {STD_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== STANDARD DEVIATION GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, theta)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for std generation...")
    
    process_results = []
    
    # Track process information (matches static_cluster_logged_mp.py)
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        process_info[dev] = {"index": i, "process_id": process_id, "status": "queued"}
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_std_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
                process_info[args[0]]["status"] = "running"
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    process_info[dev]["status"] = "completed"
                    if result["success"]:
                        action = result.get("action", "processed")
                        message = result.get("message", "")
                        master_logger.info(f"Process for dev {dev} completed successfully: "
                                         f"{action} - {message}, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process for dev {dev} failed: {result['error']}")
                        process_info[dev]["status"] = "failed"
                except TimeoutError:
                    master_logger.error(f"Process for dev {dev} timed out after {PROCESS_TIMEOUT}s")
                    process_info[dev]["status"] = "timeout"
                    process_results.append({
                        "dev": dev, "success": False, "error": "Timeout",
                        "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process for dev {dev} crashed: {str(e)}")
                    process_info[dev]["status"] = "crashed"
                    process_results.append({
                        "dev": dev, "success": False, "error": str(e),
                        "total_time": 0
                    })
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
        main_tar_name = f"experiments_data_std_N{N}_samples{samples}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        dev_tar_paths = [r.get("dev_tar_path") for r in process_results if r.get("dev_tar_path")]
        if dev_tar_paths:
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
    
    total_time = time.time() - start_time
    
    # Generate summary
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    generated_count = sum(1 for r in process_results if r.get("action") == "generated")
    skipped_count = sum(1 for r in process_results if r.get("action") == "skipped")
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Actions: {generated_count} generated, {skipped_count} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
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
