#!/usr/bin/env python3

"""
Generate Probability Distributions from Samples

This script generates probability distribution (.pkl) files from existing sample data.
It processes multiple deviation values in parallel, checking for missing or invalid
probability distribution files and creating them from the corresponding sample files.

Key Features:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if probDist files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_probdist_from_samples.py

Configuration:
    Edit the parameters section below to match your experiment setup.
"""

import os
import sys
import time
import pickle
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import traceback
import signal
import math
import numpy as np
import shutil
import glob

# ============================================================================
# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create per-dev and main tar archives
ARCHIVE_DIR = "experiments_archive"


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 20000                # System size (small for testing)
steps = N//4           # Time steps (25 for N=100)
samples = 40           # Samples per deviation (small for testing)
theta = math.pi/3      # Theta parameter for static noise

# Deviation values - TEST SET (matching generate_samples.py)
# devs = [
#     # (0,0),              # No noise
#     # (0, 0.2),           # Small noise range
#     # (0, 0.5),           # Medium noise range  
#     (0, 0.8),           # Medium noise range  
# ]

devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),           # Medium noise range  
]

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples"
PROBDIST_BASE_DIR = "experiments_data_samples_probDist"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_probdist")

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
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_probdist", "probdist_generation_master.log")

# Global shutdown flag
SHUTDOWN_REQUESTED = False

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
            warning_msg = f"{prefix} HIGH MEMORY USAGE: {memory.percent:.1f}%"
            if logger:
                logger.warning(warning_msg)
            else:
                print(f"[WARNING] {warning_msg}")
        
        if cpu_percent > 95:
            warning_msg = f"{prefix} HIGH CPU USAGE: {cpu_percent:.1f}%"
            if logger:
                logger.warning(warning_msg)
            else:
                print(f"[WARNING] {warning_msg}")
            
    except ImportError:
        msg = f"{prefix} psutil not available - cannot monitor resources"
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")
    except Exception as e:
        msg = f"{prefix} Error monitoring resources: {e}"
        if logger:
            logger.warning(msg)
        else:
            print(f"[WARNING] {msg}")

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_probdist.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_probdist")
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
    Example: experiments_data_samples/dummy_tesselation_func_static_noise/theta_1.047198/dev_min0.000_max0.000/N_300/samples_5
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

def validate_probdist_file(file_path):
    """
    Validate that a probability distribution file exists and contains valid data.
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a valid numpy array with expected properties
        if isinstance(data, np.ndarray) and data.size > 0:
            # Additional validation: check for reasonable probability values
            if np.all(data >= 0) and np.all(data <= 1) and not np.any(np.isnan(data)):
                return True
        
        return False
        
    except (pickle.PickleError, EOFError, ValueError, TypeError) as e:
        return False

def load_sample_file(file_path):
    """
    Load a sample file and return the state data.
    
    Returns:
        numpy.ndarray or None: The sample state data, or None if loading failed
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except (pickle.PickleError, EOFError, ValueError, TypeError):
        return None

# ============================================================================
# PROBABILITY DISTRIBUTION GENERATION FUNCTIONS
# ============================================================================

def generate_step_probdist(samples_dir, target_dir, step_idx, N, samples_count, logger):
    """
    Generate probability distribution for a specific step from sample files.
    Uses optimized streaming processing with incremental mean calculation to minimize memory usage.
    This matches the implementation from static_cluster_logged_mp.py.
    
    Args:
        samples_dir: Directory containing sample files
        target_dir: Directory to save probability distribution
        step_idx: Step index to process
        N: System size
        samples_count: Number of samples
        logger: Logger for this process
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Import required function
        from sqw.states import amp2prob
        
        step_dir = os.path.join(samples_dir, f"step_{step_idx}")
        if not os.path.exists(step_dir):
            logger.error(f"Step directory not found: {step_dir}")
            return False
        
        # Optimized streaming processing - load and process samples one at a time
        # This matches the memory-efficient approach from static_cluster_logged_mp.py
        mean_prob_dist = None
        valid_samples = 0
        
        for sample_idx in range(samples_count):
            # Use the actual filename format found in the directories
            filename = f"sample_{sample_idx}.pkl"
            filepath = os.path.join(step_dir, filename)
            
            if os.path.exists(filepath):
                # Load sample
                with open(filepath, "rb") as f:
                    state = pickle.load(f)
                
                # Convert to probability distribution
                prob_dist = amp2prob(state)  # |amplitude|^2
                
                # Update running mean using incremental formula
                if mean_prob_dist is None:
                    # Initialize with first sample
                    mean_prob_dist = prob_dist.copy()
                else:
                    # Incremental mean: new_mean = old_mean + (new_value - old_mean) / count
                    mean_prob_dist += (prob_dist - mean_prob_dist) / (valid_samples + 1)
                
                valid_samples += 1
                
                # Free memory immediately
                del state, prob_dist
            else:
                if sample_idx < 10:  # Only log first 10 missing files to avoid spam
                    logger.warning(f"Sample file not found: {filepath}")
        
        if valid_samples > 0:
            # Save mean probability distribution
            os.makedirs(target_dir, exist_ok=True)
            mean_filepath = os.path.join(target_dir, f"mean_step_{step_idx}.pkl")
            
            with open(mean_filepath, "wb") as f:
                pickle.dump(mean_prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Only log completion for debugging (matches static cluster behavior)
            if step_idx % 100 == 0 or step_idx < 10:
                logger.info(f"Generated probDist for step {step_idx} from {valid_samples} samples")
            return True
        else:
            logger.warning(f"No valid samples found for step {step_idx}")
            return False
        
    except Exception as e:
        logger.error(f"Error generating probDist for step {step_idx}: {str(e)}")
        return False


def mirror_existing_mean_files(samples_exp_dir, probdist_exp_dir, samples_count, logger):
    """
    If the samples directory already contains precomputed mean_step_*.pkl files
    (possibly because the samples base actually contains probDist outputs),
    mirror/copy them into the probdist target directory so downstream tools
    see the same layout as produced in experiments_data_samples_probDist.

    Returns number of files copied.
    """
    copied = 0

    # Ensure target exists
    try:
        os.makedirs(probdist_exp_dir, exist_ok=True)
    except Exception as e:
        logger.warning(f"Unable to create probdist target dir {probdist_exp_dir}: {e}")
        return 0

    # Look for mean_step_*.pkl files directly under samples_exp_dir
    patterns = [os.path.join(samples_exp_dir, 'mean_step_*.pkl'),
                os.path.join(samples_exp_dir, 'samples_*', 'mean_step_*.pkl'),
                os.path.join(samples_exp_dir, 'N_*', 'samples_*', 'mean_step_*.pkl'),
                os.path.join(samples_exp_dir, '**', 'mean_step_*.pkl')]

    found_files = set()
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            found_files.add(os.path.normpath(f))

    if not found_files:
        logger.debug(f"No precomputed mean_step_*.pkl files found to mirror from {samples_exp_dir}")
        return 0

    for src in sorted(found_files):
        # Determine destination filename: keep the same basename but place in probdist_exp_dir
        dest = os.path.join(probdist_exp_dir, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
            copied += 1
            logger.info(f"Mirrored existing probDist file: {src} -> {dest}")
        except Exception as e:
            logger.warning(f"Failed to copy {src} to {dest}: {e}")

    # If we copied files, ensure permissions and return count
    return copied


def validate_samples_presence(samples_exp_dir, steps, samples_count, logger):
    """
    Check that expected sample files exist for all steps and samples_count.
    Returns True if all expected sample files are present and loadable, False otherwise.
    """
    missing = []
    invalid = []

    for step_idx in range(steps):
        step_dir = os.path.join(samples_exp_dir, f"step_{step_idx}")
        if not os.path.isdir(step_dir):
            missing.append(step_dir)
            continue

        for sample_idx in range(samples_count):
            sample_file = os.path.join(step_dir, f"sample_{sample_idx}.pkl")
            if not os.path.exists(sample_file):
                missing.append(sample_file)
            else:
                # quick load check
                data = load_sample_file(sample_file)
                if data is None:
                    invalid.append(sample_file)

    if missing or invalid:
        if missing:
            logger.error(f"Missing sample files or step dirs: {len(missing)} items. Example: {missing[:3]}")
        if invalid:
            logger.error(f"Invalid sample files (failed to load): {len(invalid)} items. Example: {invalid[:3]}")
        return False

    logger.info(f"All expected sample files present and valid: steps={steps}, samples={samples_count}")
    return True

def generate_probdist_for_dev(dev_args):
    """
    Worker function to generate probability distributions for a single deviation value.
    dev_args: (dev, process_id, N, steps, samples, theta, shutdown_flag, source_base_dir, target_base_dir)
    """
    dev, process_id, N, steps, samples_count, theta_param, shutdown_flag, source_base_dir, target_base_dir = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"=== PROBDIST GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        
        dev_start_time = time.time()
        
        # Handle deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            has_noise = max_val > 0
        else:
            has_noise = dev > 0
        
        # Get source and target directories using smart_loading_static
        from smart_loading_static import find_experiment_dir_flexible, get_experiment_dir
        
        noise_params = [dev]
        samples_exp_dir, found_format = find_experiment_dir_flexible(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=source_base_dir, theta=theta_param
        )
        
        # Check if the actual samples are in a samples_X subdirectory
        potential_samples_dir = os.path.join(samples_exp_dir, f"samples_{samples_count}")
        if os.path.exists(potential_samples_dir) and os.path.isdir(potential_samples_dir):
            # Check if this subdirectory contains step directories
            sample_step_dirs = [d for d in os.listdir(potential_samples_dir) 
                              if os.path.isdir(os.path.join(potential_samples_dir, d)) and d.startswith('step_')]
            if sample_step_dirs:
                samples_exp_dir = potential_samples_dir
                logger.info(f"Found samples in subdirectory: samples_{samples_count}")
        
        probdist_exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=target_base_dir, theta=theta_param, samples=samples_count
        )
        logger.info(f"Samples source: {samples_exp_dir}")
        logger.info(f"ProbDist target: {probdist_exp_dir}")
        logger.info(f"Source format: {found_format}")
        # If there are precomputed mean_step_*.pkl files in the samples tree,
        # mirror them into the probdist target so the output layout matches
        # the expected experiments_data_samples_probDist structure.
        try:
            copied = mirror_existing_mean_files(samples_exp_dir, probdist_exp_dir, samples_count, logger)
            if copied > 0:
                logger.info(f"Mirrored {copied} precomputed mean files to probdist target")
        except Exception as e:
            logger.warning(f"Error while attempting to mirror existing mean files: {e}")

        # Validate presence and loadability of sample files for this experiment
        valid_samples = validate_samples_presence(samples_exp_dir, steps, samples_count, logger)
        if not valid_samples:
            # Signal shutdown across processes
            try:
                shutdown_flag.value = True
            except Exception:
                pass
            logger.error(f"Critical: Missing or invalid sample files for {samples_exp_dir}. Signalling shutdown.")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": "Missing or invalid sample files; aborting.",
                "processed_steps": 0, "total_steps": steps,
                "skipped_steps": 0, "generated_steps": 0,
                "log_file": log_file, "total_time": 0
            }
        
        # Check if samples directory exists. If missing, create a placeholder
        # directory (so paths are consistent) and return a clear message that
        # samples must be generated first. Creating an empty samples folder
        # won't allow generation of probDist files, but prevents hard failures
        # elsewhere and gives a friendly next step.
        if not os.path.exists(samples_exp_dir):
            try:
                os.makedirs(samples_exp_dir, exist_ok=True)
                logger.warning(f"Samples directory not found; created placeholder: {samples_exp_dir}")
            except Exception as e:
                logger.error(f"Unable to create placeholder samples directory: {samples_exp_dir}: {e}")
                return {
                    "dev": dev, "process_id": process_id, "success": False,
                    "error": f"Samples directory not found and could not be created: {samples_exp_dir}",
                    "processed_steps": 0, "total_steps": steps,
                    "log_file": log_file, "total_time": 0
                }

            # Inform the caller that samples are missing and must be generated first
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": f"Samples directory not found: {samples_exp_dir} (placeholder created). Run generate_samples.py for this deviation.",
                "processed_steps": 0, "total_steps": steps,
                "log_file": log_file, "total_time": 0
            }
        
        # Process each step with enhanced monitoring (matches static_cluster_logged_mp.py)
        processed_steps = 0
        skipped_steps = 0
        generated_steps = 0
        last_log_time = time.time()
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        logger.info(f"Processing {steps} steps for deviation {dev_str}")
        
        for step_idx in range(steps):
            global SHUTDOWN_REQUESTED
            if SHUTDOWN_REQUESTED:
                logger.warning(f"Shutdown requested, stopping at step {step_idx}")
                break
            
            # Log progress more frequently and monitor resources
            current_time = time.time()
            # Log every 100 steps, but only log time-based updates if it's been more than 5 minutes
            # AND we're not already logging for the 100-step interval
            should_log_progress = (step_idx % 100 == 0)
            should_log_resources = (current_time - last_log_time >= 300)  # Every 5 minutes
            
            if should_log_progress:
                log_progress_update("PROBDIST", step_idx + 1, steps, dev_start_time, logger)
            
            if should_log_resources:
                log_system_resources(logger, "[WORKER]")
                last_log_time = current_time
            
            # Check if probDist file exists and is valid
            probdist_file = os.path.join(probdist_exp_dir, f"mean_step_{step_idx}.pkl")
            
            if validate_probdist_file(probdist_file):
                skipped_steps += 1
                if step_idx % 100 == 0:
                    logger.info(f"    Step {step_idx+1}/{steps} already exists, skipping")
            else:
                # Only log generation for debugging (matches static cluster behavior)
                if step_idx % 100 == 0:
                    logger.info(f"    Step {step_idx+1}/{steps} processing... (processed: {processed_steps})")
                    
                if generate_step_probdist(samples_exp_dir, probdist_exp_dir, step_idx, N, samples_count, logger):
                    generated_steps += 1
                else:
                    logger.error(f"Failed to generate probDist for step {step_idx}")
            
            processed_steps += 1
            
            # Force garbage collection periodically to keep memory usage low
            if step_idx % 50 == 0:
                import gc
                gc.collect()
        
        dev_time = time.time() - dev_start_time
        
        logger.info(f"=== PROBDIST GENERATION COMPLETED ===")
        logger.info(f"Processed: {processed_steps}/{steps} steps")
        logger.info(f"Skipped (already valid): {skipped_steps}")
        logger.info(f"Generated: {generated_steps}")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        # Archive this dev's probdist directory if requested

        dev_tar_path = None
        if CREATE_TAR:
            import tarfile
            os.makedirs(ARCHIVE_DIR, exist_ok=True)
            # Format dev string for filename
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                min_val, max_val = dev
                devstr = f"min{min_val:.3f}_max{max_val:.3f}"
            else:
                devstr = f"{float(dev):.3f}"
            # Find the theta_ folder containing this probdist_exp_dir
            theta_dir = os.path.dirname(os.path.dirname(probdist_exp_dir))
            theta_folder_name = os.path.basename(theta_dir)
            dev_folder_name = os.path.basename(os.path.dirname(probdist_exp_dir))
            # Archive only the dev folder inside theta_ (not the whole theta_ folder)
            dev_dir = os.path.dirname(probdist_exp_dir)
            dev_tar_path = os.path.join(ARCHIVE_DIR, f"probdist_{theta_folder_name}_{devstr}.tar")
            with tarfile.open(dev_tar_path, "w") as tar:
                tar.add(dev_dir, arcname=os.path.join(theta_folder_name, dev_folder_name))
            logger.info(f"Created temporary archive: {dev_tar_path}")

        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "error": None,
            "processed_steps": processed_steps,
            "total_steps": steps,
            "skipped_steps": skipped_steps,
            "generated_steps": generated_steps,
            "log_file": log_file,
            "total_time": dev_time,
            "dev_tar_path": dev_tar_path
        }
        
    except Exception as e:
        error_msg = f"Error in probDist generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": False,
            "error": error_msg,
            "processed_steps": 0,
            "total_steps": steps,
            "skipped_steps": 0,
            "generated_steps": 0,
            "log_file": log_file,
            "total_time": 0
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def create_probability_distributions_multiprocess(
    tesselation_func,
    N,
    steps,
    devs,
    samples,
    source_base_dir="experiments_data_samples",
    target_base_dir="experiments_data_samples_probDist",
    noise_type="static_noise",
    theta=None,
    use_multiprocess=True,
    max_processes=None,
    logger=None
):
    """
    Create probability distributions using multiprocessing for parallel computation.
    Each deviation is processed in a separate process for maximum efficiency.
    This function matches the interface from static_cluster_logged_mp.py.
    
    Args:
        tesselation_func: Function to create tesselation (dummy for static noise)
        N: System size
        steps: Number of time steps
        devs: List of deviation values
        samples: Number of samples per deviation
        source_base_dir: Base directory containing sample data
        target_base_dir: Base directory to save probability distributions
        noise_type: Type of noise ("static_noise")
        theta: Theta parameter for static noise
        use_multiprocess: Whether to use multiprocessing
        max_processes: Maximum number of processes (None = auto-detect)
        logger: Optional logger for logging operations
    
    Returns:
        List of results from each process
    """
    def log_and_print(message, level="info"):
        """Helper function to log messages (cluster-safe, no print)"""
        if logger:
            if level == "info":
                logger.info(message.replace("[PROBDIST] ", "").replace("[WARNING] ", "").replace("[OK] ", "").replace("[ERROR] ", ""))
            elif level == "warning":
                logger.warning(message.replace("[WARNING] ", "").replace("[PROBDIST] ", ""))
            elif level == "error":
                logger.error(message.replace("[ERROR] ", "").replace("[PROBDIST] ", ""))
        else:
            print(message)
    
    log_and_print(f"[PROBDIST] Creating probability distributions for {len(devs)} devs...")
    start_time = time.time()
    
    # Check if multiprocessing should be used
    if not use_multiprocess or len(devs) <= 1:
        log_and_print(f"[PROBDIST] Using single-process mode for {len(devs)} deviations")
        
        # Fall back to sequential processing
        for dev in devs:
            # Process each deviation sequentially
            result = generate_probdist_for_dev((dev, 0, N, steps, samples, theta, None))
            if not result["success"]:
                log_and_print(f"[ERROR] Failed to process deviation {dev}: {result['error']}", "error")
        
        total_time = time.time() - start_time
        log_and_print(f"[OK] Probability distributions created in {total_time:.1f}s (sequential)")
        return []
    
    # Multiprocessing approach
    if max_processes is None:
        max_processes = min(len(devs), mp.cpu_count())
    
    log_and_print(f"[PROBDIST] Using multiprocess mode with {max_processes} processes for {len(devs)} deviations")
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        # Create shared shutdown flag for this run
        manager = mp.Manager()
        shutdown_flag = manager.Value('b', False)
        args = (dev, process_id, N, steps, samples, theta, shutdown_flag, source_base_dir, target_base_dir)
        process_args.append(args)
    
    # Use the same multiprocessing logic as the main function
    # (This code would be shared with the main() function)
    
    process_results = []
    
    # Track process information
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"{float(dev):.3f}"
        
        process_info[dev] = {
            "process_id": process_id,
            "log_file": f"process_dev_{dev_str}_probdist.log",
            "start_time": None,
            "end_time": None,
            "status": "pending"
        }
    
    try:
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all jobs
            future_to_dev = {}
            for args in process_args:
                dev = args[0]
                future = executor.submit(generate_probdist_for_dev, args)
                future_to_dev[future] = dev
                process_info[dev]["start_time"] = time.time()
                process_info[dev]["status"] = "running"
            
            # Collect results
            for future in as_completed(future_to_dev, timeout=max(3600, len(devs) * 1800)):
                dev = future_to_dev[future]
                try:
                    result = future.result(timeout=10)
                    process_results.append(result)
                    process_info[dev]["status"] = "completed" if result["success"] else "failed"
                except Exception as e:
                    log_and_print(f"[ERROR] Process for dev {dev} failed: {str(e)}", "error")
                    process_results.append({
                        "dev": dev, "success": False, "error": str(e),
                        "processed_steps": 0, "total_steps": steps,
                        "total_time": 0
                    })
                    process_info[dev]["status"] = "crashed"
                    
    except Exception as e:
        log_and_print(f"[ERROR] Critical error in multiprocessing: {str(e)}", "error")
        raise
    
    total_time = time.time() - start_time
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    
    log_and_print(f"[OK] Probability distributions multiprocessing completed in {total_time:.1f}s")
    log_and_print(f"[OK] Results: {successful_processes} successful, {failed_processes} failed processes")
    
    return process_results

def main():
    """Main execution function for probability distribution generation."""
    
    print("=== PROBABILITY DISTRIBUTION GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"Samples source: {SAMPLES_BASE_DIR}")
    print(f"ProbDist target: {PROBDIST_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== PROBABILITY DISTRIBUTION GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()

    # Pre-flight: ensure probdist directories exist and mirror any precomputed mean files
    master_logger.info("Preparing probdist directory structure and mirroring existing mean files (pre-flight)")
    for dev in devs:
        try:
            # Determine if noise present
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                has_noise = dev[1] > 0
            else:
                has_noise = float(dev) > 0

            samples_exp_dir, found_format = find_experiment_dir_flexible(
                dummy_tesselation_func, has_noise, N, noise_params=[dev], noise_type="static_noise", base_dir=SAMPLES_BASE_DIR, theta=theta
            )

            probdist_exp_dir = get_experiment_dir(
                dummy_tesselation_func, has_noise, N, noise_params=[dev], noise_type="static_noise", base_dir=PROBDIST_BASE_DIR, theta=theta, samples=samples
            )

            # Create target dir and mirror any mean files from samples tree
            os.makedirs(probdist_exp_dir, exist_ok=True)
            copied = mirror_existing_mean_files(samples_exp_dir, probdist_exp_dir, samples, master_logger)
            if copied > 0:
                master_logger.info(f"Pre-flight: mirrored {copied} mean files for dev {dev} into {probdist_exp_dir}")

        except Exception as e:
            master_logger.warning(f"Pre-flight error preparing probdist for dev {dev}: {e}")
    
    # Prepare a shared shutdown flag for workers
    manager = mp.Manager()
    shutdown_flag = manager.Value('b', False)

    # Prepare arguments for each process (include shutdown_flag and base directories)
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, theta, shutdown_flag, SAMPLES_BASE_DIR, PROBDIST_BASE_DIR)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for probDist generation...")
    
    process_results = []
    
    # Track process information (matches static_cluster_logged_mp.py)
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        # Format dev for filename
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"{float(dev):.3f}"
        
        log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_probdist.log")
        process_info[dev] = {
            "process_id": process_id,
            "log_file": log_file,
            "start_time": None,
            "end_time": None,
            "status": "pending"
        }
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all jobs
            future_to_dev = {}
            for args in process_args:
                dev = args[0]
                future = executor.submit(generate_probdist_for_dev, args)
                future_to_dev[future] = dev
                process_info[dev]["start_time"] = time.time()
                process_info[dev]["status"] = "running"
                
                # Format dev for display
                if isinstance(dev, (tuple, list)) and len(dev) == 2:
                    dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
                else:
                    dev_str = f"{float(dev):.3f}"
                master_logger.info(f"Process launched for dev={dev_str}")
            
            # Collect results as they complete with timeout and resource monitoring
            completed = 0
            timeout_start = time.time()
            last_progress_time = timeout_start
            master_logger.info(f"Waiting for {len(future_to_dev)} processes with {PROCESS_TIMEOUT/3600:.1f}h timeout...")
            log_system_resources(master_logger, "[MASTER]")
            
            try:
                for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT):
                    # Check for shutdown signal
                    if SHUTDOWN_REQUESTED:
                        master_logger.warning("Graceful shutdown requested, cancelling remaining processes...")
                        for remaining_future in future_to_dev:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    dev = future_to_dev[future]
                    process_info[dev]["end_time"] = time.time()
                    
                    try:
                        result = future.result(timeout=10)  # Short timeout since future is already done
                        process_results.append(result)
                        process_info[dev]["status"] = "completed" if result["success"] else "failed"
                        
                        # Format dev for display
                        if isinstance(dev, (tuple, list)) and len(dev) == 2:
                            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
                        else:
                            dev_str = f"{float(dev):.3f}"
                        
                        if result["success"]:
                            master_logger.info(f"[SUCCESS] Dev {dev_str}: {result['generated_steps']} generated, "
                                             f"{result['skipped_steps']} skipped, {result['total_time']:.1f}s")
                        else:
                            master_logger.error(f"[FAILED] Dev {dev_str}: {result['error']}")
                            
                    except TimeoutError:
                        error_msg = f"Process result timeout after completion for dev {dev}"
                        master_logger.error(error_msg)
                        process_results.append({
                            "dev": dev, "process_id": process_info[dev]["process_id"], "success": False,
                            "error": error_msg, "processed_steps": 0, "total_steps": steps,
                            "skipped_steps": 0, "generated_steps": 0,
                            "log_file": process_info[dev]["log_file"], "total_time": 0
                        })
                        process_info[dev]["status"] = "timeout"
                        
                    except Exception as e:
                        error_msg = f"Process exception for dev {dev}: {str(e)}"
                        master_logger.error(error_msg)
                        process_results.append({
                            "dev": dev, "process_id": process_info[dev]["process_id"], "success": False,
                            "error": error_msg, "processed_steps": 0, "total_steps": steps,
                            "skipped_steps": 0, "generated_steps": 0,
                            "log_file": process_info[dev]["log_file"], "total_time": 0
                        })
                        process_info[dev]["status"] = "crashed"
                    
                    completed += 1
                    
                    # Progress and resource monitoring
                    current_time = time.time()
                    if completed % max(1, len(devs) // 10) == 0 or current_time - last_progress_time >= 300:
                        log_progress_update("PROCESSES", completed, len(devs), timeout_start, master_logger)
                        log_system_resources(master_logger, "[MASTER]")
                        last_progress_time = current_time
                    
                    # Check shared shutdown flag
                    try:
                        if shutdown_flag.value:
                            master_logger.warning("Shutdown flag detected, cancelling remaining processes...")
                            for remaining_future in future_to_dev:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            break
                    except Exception:
                        pass
                        
            except TimeoutError:
                master_logger.error(f"Overall timeout after {PROCESS_TIMEOUT}s - some processes may still be running")
                # Handle timeout - mark remaining futures as timed out
                for future, dev in future_to_dev.items():
                    if not future.done():
                        process_info[dev]["status"] = "timeout"
                        process_results.append({
                            "dev": dev, "process_id": process_info[dev]["process_id"], "success": False,
                            "error": "Overall timeout", "processed_steps": 0, "total_steps": steps,
                            "skipped_steps": 0, "generated_steps": 0,
                            "log_file": process_info[dev]["log_file"], "total_time": PROCESS_TIMEOUT
                        })
                        
    except KeyboardInterrupt:
        master_logger.warning("Interrupted by user")
        print("\n[INTERRUPT] Gracefully shutting down processes...")
    except Exception as e:
        master_logger.error(f"Critical error in multiprocessing: {str(e)}")
        master_logger.error(traceback.format_exc())
        raise

    # === MAIN ARCHIVE CREATION ===
    if CREATE_TAR:
        import tarfile
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        main_tar_name = f"experiments_data_probDist_N{N}_samples{samples}_{timestamp}.tar"
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
    total_generated = sum(r.get("generated_steps", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_steps", 0) for r in process_results)
    total_processed = sum(r.get("processed_steps", 0) for r in process_results)
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Steps: {total_processed} processed, {total_generated} generated, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Steps: {total_processed} processed, {total_generated} generated, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Dev {result['dev']}: SUCCESS - "
                             f"Generated: {result.get('generated_steps', 0)}, "
                             f"Skipped: {result.get('skipped_steps', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Dev {result['dev']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "total_generated": total_generated,
        "total_skipped": total_skipped,
        "total_processed": total_processed,
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
