#!/usr/bin/env python3

"""
Generate Survival Probability Data from Probability Distributions - Linspace Version

This script generates survival probability data from existing probability distribution files
created by the linspace probdist generation script. It processes multiple deviation values
using configurable multiprocessing, calculating survival probabilities for configurable
node ranges and saving the results for later plotting.

Key Features:
- Configurable number of processes (each handling multiple deviation values)
- Configurable survival probability ranges (centered and full system)
- Smart file validation (checks if survival files exist and have valid data)
- Automatic directory structure handling with static_noise_linspace
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_survival_probability_linspace.py

Configuration:
    Edit the parameters section below to match your experiment setup.
    Adjust SURVIVAL_RANGES to customize the node ranges for probability calculation.
"""

import os
import sys
import time
import math
import pickle
import logging
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import traceback
import tarfile
import signal

# ============================================================================
# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create main tar archives
ARCHIVE_DIR = "experiments_archive_linspace"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 4000                 # System size (reduced from 20000)
steps = N//4             # Time steps
samples = 20             # Samples per deviation
theta = math.pi/3        # Theta parameter for static noise

# Deviation values - LINSPACE BETWEEN 0.6 AND 1.0 WITH 100 VALUES
DEV_MIN = 0.6
DEV_MAX = 1.0
DEV_COUNT = 100
devs = [(0, dev) for dev in np.linspace(DEV_MIN, DEV_MAX, DEV_COUNT)]

# Multiprocessing configuration
NUM_PROCESSES = 5        # Number of processes to use (CONFIGURABLE)

# Survival probability ranges configuration
SURVIVAL_RANGES = {
    "center": {
        "description": "Central nodes (N//4 around center)",
        "range_func": lambda N: range(N//2 - N//8, N//2 + N//8 + 1)
    },
    "range_80_80": {
        "description": "Nodes from -80 to +80 relative to center",
        "range_func": lambda N: range(max(0, N//2 - 80), min(N, N//2 + 80 + 1))
    },
    "system": {
        "description": "Full system",
        "range_func": lambda N: range(N)
    }
}

# Directory configuration
PROBDIST_BASE_DIR = "experiments_data_samples_linspace_probDist"
SURVIVAL_BASE_DIR = "experiments_data_samples_linspace_survival"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_survival_linspace")

# Timeout configuration - Scale with problem size
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {NUM_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_survival_linspace", "survival_generation_master.log")

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

# ============================================================================
# SYSTEM MONITORING AND LOGGING UTILITIES
# ============================================================================

def log_system_resources(logger=None, prefix="[SYSTEM]"):
    """Log current system resource usage"""
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

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_process_logging(process_id, dev_chunk, theta=None):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format deviation chunk for filename
    if len(dev_chunk) > 0:
        first_dev = dev_chunk[0]
        last_dev = dev_chunk[-1]
        # Extract the actual deviation values from tuples
        first_val = first_dev[1] if isinstance(first_dev, (tuple, list)) and len(first_dev) > 1 else first_dev
        last_val = last_dev[1] if isinstance(last_dev, (tuple, list)) and len(last_dev) > 1 else last_dev
        dev_str = f"dev{first_val:.3f}_to_{last_val:.3f}"
    else:
        dev_str = "empty_chunk"
    
    # Format theta for filename
    if theta is not None:
        theta_str = f"_theta{theta:.6f}"
    else:
        theta_str = ""
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_{process_id}_{dev_str}{theta_str}_survival.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"process_{process_id}_survival")
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
    file_formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [PROCESS:%(name)s] %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('[PROCESS:%(name)s] %(message)s')
    
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
# DEVIATION CHUNKING FUNCTIONS
# ============================================================================

def chunk_deviations(deviations, num_processes):
    """
    Split deviations into chunks for multiprocessing.
    
    Args:
        deviations: List of deviation values
        num_processes: Number of processes to split across
    
    Returns:
        list: List of deviation chunks, one per process
    """
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")
    
    # Calculate chunk size
    chunk_size = len(deviations) // num_processes
    remainder = len(deviations) % num_processes
    
    chunks = []
    start_idx = 0
    
    for i in range(num_processes):
        # Distribute remainder across first few chunks
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk = deviations[start_idx:end_idx]
        chunks.append(chunk)
        
        start_idx = end_idx
    
    return chunks

# ============================================================================
# FILE VALIDATION FUNCTIONS
# ============================================================================

def validate_survival_file(file_path):
    """
    Validate that a survival probability file exists and contains valid data.
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a valid dictionary with expected keys
        if isinstance(data, dict) and len(data) > 0:
            # Additional validation: check for reasonable probability values
            for range_name, range_data in data.items():
                if isinstance(range_data, (list, np.ndarray)) and len(range_data) > 0:
                    # Check if values are in valid probability range [0, 1]
                    if all(0 <= x <= 1 for x in range_data if x is not None):
                        return True
        
        return False
        
    except (pickle.PickleError, EOFError, ValueError, TypeError) as e:
        return False

# ============================================================================
# DIRECTORY AND FILE MANAGEMENT
# ============================================================================

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
# SURVIVAL PROBABILITY CALCULATION FUNCTIONS
# ============================================================================

def calculate_survival_probabilities_for_range(prob_distributions, node_range, logger=None):
    """
    Calculate survival probabilities for a specific range of nodes.
    
    Args:
        prob_distributions: List of probability distributions for each time step
        node_range: Range of nodes to sum probabilities over
        logger: Optional logger for debug output
    
    Returns:
        list: Survival probabilities for each time step
    """
    survival_probs = []
    node_indices = list(node_range)
    
    if logger:
        logger.debug(f"Calculating survival for nodes: {min(node_indices)} to {max(node_indices)} ({len(node_indices)} nodes)")
    
    for step_idx, prob_dist in enumerate(prob_distributions):
        if prob_dist is not None and len(prob_dist) > 0:
            try:
                # Sum probabilities over the specified range
                survival_prob = sum(prob_dist[i] for i in node_indices if 0 <= i < len(prob_dist))
                survival_probs.append(survival_prob)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Error calculating survival for step {step_idx}: {e}")
                survival_probs.append(None)
        else:
            survival_probs.append(None)
    
    return survival_probs

def generate_survival_probability_data(prob_distributions, N, logger=None):
    """
    Generate survival probability data for all configured ranges.
    
    Args:
        prob_distributions: List of probability distributions for each time step
        N: System size
        logger: Optional logger for output
    
    Returns:
        dict: Dictionary mapping range names to survival probability lists
    """
    survival_data = {}
    
    for range_name, range_config in SURVIVAL_RANGES.items():
        try:
            if logger:
                logger.info(f"Calculating survival probabilities for range '{range_name}': {range_config['description']}")
            
            # Get the node range for this configuration
            node_range = range_config['range_func'](N)
            
            # Calculate survival probabilities
            survival_probs = calculate_survival_probabilities_for_range(
                prob_distributions, node_range, logger
            )
            
            survival_data[range_name] = survival_probs
            
            valid_count = sum(1 for p in survival_probs if p is not None)
            if logger:
                logger.info(f"Range '{range_name}': {valid_count}/{len(survival_probs)} valid values")
                if valid_count > 0:
                    final_prob = [p for p in survival_probs if p is not None][-1]
                    # Convert to scalar if it's a numpy array
                    if hasattr(final_prob, 'item'):
                        final_prob = final_prob.item()
                    logger.info(f"Range '{range_name}' final survival probability: {final_prob:.6f}")
            
        except Exception as e:
            if logger:
                logger.error(f"Error calculating survival for range '{range_name}': {e}")
            survival_data[range_name] = [None] * len(prob_distributions)
    
    return survival_data

def generate_survival_for_dev_chunk(chunk_args):
    """
    Worker function to generate survival probability data for a chunk of deviation values.
    
    Args:
        chunk_args: Tuple containing (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, probdist_base_dir, survival_base_dir)
    
    Returns:
        dict: Results from the survival probability generation process
    """
    dev_chunk, process_id, N, steps, samples_count, theta_param, shutdown_flag, probdist_base_dir, survival_base_dir = chunk_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    from smart_loading_static import format_theta_for_directory
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(process_id, dev_chunk, theta_param)
    
    try:
        logger.info(f"=== SURVIVAL PROBABILITY GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Process ID: {process_id}")
        logger.info(f"Deviation chunk size: {len(dev_chunk)}")
        if len(dev_chunk) > 0:
            first_dev = dev_chunk[0]
            last_dev = dev_chunk[-1]
            # Extract the actual deviation values from tuples
            first_val = first_dev[1] if isinstance(first_dev, (tuple, list)) and len(first_dev) > 1 else first_dev
            last_val = last_dev[1] if isinstance(last_dev, (tuple, list)) and len(last_dev) > 1 else last_dev
            logger.info(f"Deviation range: {first_val:.4f} to {last_val:.4f}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        logger.info(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
        
        chunk_start_time = time.time()
        chunk_generated_files = 0
        chunk_skipped_files = 0
        processed_devs = 0
        
        # Process each deviation in the chunk
        for dev in dev_chunk:
            dev_start_time = time.time()
            logger.info(f"Processing deviation {processed_devs+1}/{len(dev_chunk)}: {dev}")
            
            # Format deviation for directory name
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_min, dev_max = dev
                dev_folder = f"dev_min{dev_min:.3f}_max{dev_max:.3f}"
            else:
                dev_folder = f"dev_{dev:.3f}"
            
            # Create directory structure for probdist and survival
            theta_folder = format_theta_for_directory(theta_param)
            
            # ProbDist source directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            probdist_exp_dir = os.path.join(probdist_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            # Survival target directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            survival_exp_dir = os.path.join(survival_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            logger.info(f"ProbDist source: {probdist_exp_dir}")
            logger.info(f"Survival target: {survival_exp_dir}")
            
            # Check if probdist directory exists
            if not os.path.exists(probdist_exp_dir):
                logger.error(f"ProbDist directory not found: {probdist_exp_dir}")
                continue
            
            # Check if survival file already exists and is valid
            os.makedirs(survival_exp_dir, exist_ok=True)
            survival_file = os.path.join(survival_exp_dir, "survival_vs_time.pkl")
            
            if validate_survival_file(survival_file):
                logger.info(f"Valid survival file already exists: {survival_file}")
                chunk_skipped_files += 1
            else:
                # Load probability distributions
                prob_distributions = load_probability_distributions_for_dev(probdist_exp_dir, N, steps, logger)
                if not prob_distributions or len(prob_distributions) == 0:
                    logger.error(f"No probability distributions loaded for dev {dev}")
                    continue
                
                # Generate survival probability data
                logger.info(f"Generating survival probability data for {len(prob_distributions)} time steps...")
                survival_data = generate_survival_probability_data(prob_distributions, N, logger)
                
                valid_ranges = sum(1 for range_name, range_data in survival_data.items() 
                                  if any(p is not None for p in range_data))
                
                if valid_ranges == 0:
                    logger.error(f"No valid survival probabilities calculated for dev {dev}")
                    continue
                
                # Save survival probability data
                with open(survival_file, 'wb') as f:
                    pickle.dump(survival_data, f)
                logger.info(f"Survival probability data saved to: {survival_file}")
                chunk_generated_files += 1
            
            processed_devs += 1
            dev_time = time.time() - dev_start_time
            logger.info(f"Deviation {dev} completed in {dev_time:.1f}s")
            
            # Garbage collection
            gc.collect()
            
            # Check for shutdown signal
            if shutdown_flag.value:
                logger.warning("Shutdown signal received, stopping processing")
                break
        
        chunk_time = time.time() - chunk_start_time
        logger.info(f"=== SURVIVAL PROBABILITY GENERATION COMPLETED ===")
        logger.info(f"Process {process_id}: {processed_devs} deviations processed")
        logger.info(f"Generated: {chunk_generated_files}, Skipped: {chunk_skipped_files}")
        logger.info(f"Total time: {chunk_time:.1f}s")
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": processed_devs,
            "generated_files": chunk_generated_files,
            "skipped_files": chunk_skipped_files,
            "total_time": chunk_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error in survival generation for process {process_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": 0,
            "generated_files": 0,
            "skipped_files": 0,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for survival probability generation."""
    
    print("=== SURVIVAL PROBABILITY GENERATION - LINSPACE VERSION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    print(f"Total deviations: {len(devs)}")
    print(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
    print(f"Processes: {NUM_PROCESSES}")
    print(f"ProbDist source: {PROBDIST_BASE_DIR}")
    print(f"Survival target: {SURVIVAL_BASE_DIR}")
    print("=" * 60)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== SURVIVAL PROBABILITY GENERATION - LINSPACE VERSION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    master_logger.info(f"Total deviations: {len(devs)}")
    master_logger.info(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
    master_logger.info(f"Processes: {NUM_PROCESSES}")
    
    start_time = time.time()
    
    # Split deviations into chunks for processes
    dev_chunks = chunk_deviations(devs, NUM_PROCESSES)
    
    # Log chunk distribution
    master_logger.info("DEVIATION CHUNK DISTRIBUTION:")
    for i, chunk in enumerate(dev_chunks):
        if len(chunk) > 0:
            first_dev = chunk[0]
            last_dev = chunk[-1]
            # Extract the actual deviation values from tuples
            first_val = first_dev[1] if isinstance(first_dev, (tuple, list)) and len(first_dev) > 1 else first_dev
            last_val = last_dev[1] if isinstance(last_dev, (tuple, list)) and len(last_dev) > 1 else last_dev
            master_logger.info(f"  Process {i}: {len(chunk)} deviations ({first_val:.4f} to {last_val:.4f})")
        else:
            master_logger.info(f"  Process {i}: 0 deviations (empty)")
    
    # Prepare a shared shutdown flag for workers
    manager = mp.Manager()
    shutdown_flag = manager.Value('b', False)
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev_chunk in enumerate(dev_chunks):
        args = (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, PROBDIST_BASE_DIR, SURVIVAL_BASE_DIR)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for survival generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:
            # Submit all processes
            future_to_process = {}
            for args in process_args:
                future = executor.submit(generate_survival_for_dev_chunk, args)
                future_to_process[future] = args[1]  # process_id
            
            # Collect results with timeout handling
            for future in as_completed(future_to_process, timeout=PROCESS_TIMEOUT * len(process_args)):
                process_id = future_to_process[future]
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    if result["success"]:
                        master_logger.info(f"Process {process_id} completed successfully: "
                                         f"Generated: {result['generated_files']}, "
                                         f"Skipped: {result['skipped_files']}, "
                                         f"Time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process {process_id} failed: {result['error']}")
                except TimeoutError:
                    master_logger.error(f"Process {process_id} timed out after {PROCESS_TIMEOUT}s")
                    shutdown_flag.value = True
                    process_results.append({
                        "process_id": process_id, "success": False, "error": "Timeout",
                        "total_time": PROCESS_TIMEOUT, "generated_files": 0, "skipped_files": 0
                    })
                except Exception as e:
                    master_logger.error(f"Process {process_id} crashed: {str(e)}")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": str(e),
                        "total_time": 0, "generated_files": 0, "skipped_files": 0
                    })
    except KeyboardInterrupt:
        master_logger.warning("Interrupted by user")
        shutdown_flag.value = True
        print("\n[INTERRUPT] Gracefully shutting down processes...")
    except Exception as e:
        master_logger.error(f"Critical error in multiprocessing: {str(e)}")
        raise

    # Calculate total statistics
    total_processed_devs = sum(r.get("processed_devs", 0) for r in process_results)
    total_generated = sum(r.get("generated_files", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_files", 0) for r in process_results)
    
    # === MAIN ARCHIVE CREATION ===
    if CREATE_TAR:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        main_tar_name = f"experiments_data_survival_linspace_N{N}_samples{samples}_theta{theta:.6f}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        
        print(f"Creating archive: {main_tar_path}")
        master_logger.info(f"Creating archive: {main_tar_path}")
        
        try:
            with tarfile.open(main_tar_path, "w") as tar:
                tar.add(SURVIVAL_BASE_DIR, arcname=os.path.basename(SURVIVAL_BASE_DIR))
            print(f"Created main archive: {main_tar_path}")
            master_logger.info(f"Created main archive: {main_tar_path}")
        except Exception as e:
            master_logger.error(f"Failed to create archive: {e}")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Generate final summary with enhanced validation
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    print(f"Actions: {total_generated} generated, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    master_logger.info(f"Actions: {total_generated} generated, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Process {result['process_id']}: SUCCESS - "
                             f"Devs: {result.get('processed_devs', 0)}, "
                             f"Generated: {result.get('generated_files', 0)}, "
                             f"Skipped: {result.get('skipped_files', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Process {result['process_id']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "total_generated": total_generated,
        "total_skipped": total_skipped,
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
