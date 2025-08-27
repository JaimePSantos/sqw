#!/usr/bin/env python3

"""
Generate Standard Deviation Data from Probability Distributions - Linspace Version

This script generates standard deviation data from existing probability distribution files
created by the linspace probdist generation script. It processes multiple deviation values
using configurable multiprocessing, calculating standard deviations across time for each
deviation and saving the results for later plotting.

Key Features:
- Configurable number of processes (each handling multiple deviation values)
- Smart file validation (checks if std files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_std_from_probdist_linspace.py

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

CREATE_TAR = True  # If True, create per-chunk and main tar archives
ARCHIVE_DIR = "experiments_archive_linspace_std"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 100                 # System size (matching generate_samples_linspace.py)
steps = N//4             # Time steps
samples = 2              # Samples per deviation (matching generate_samples_linspace.py)
theta = math.pi/3        # Theta parameter for static noise

# Deviation values - LINSPACE BETWEEN 0.6 AND 1.0 WITH 100 VALUES (matching generate_samples_linspace.py)
DEV_MIN = 0.6
DEV_MAX = 1.0
DEV_COUNT = 20
devs = [(0, dev) for dev in np.linspace(DEV_MIN, DEV_MAX, DEV_COUNT)]

# Multiprocessing configuration
NUM_PROCESSES = 5        # Number of processes to use (CONFIGURABLE)

# Timeout configuration - Scale with problem size
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {NUM_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# Directory configuration
PROBDIST_BASE_DIR = "experiments_data_samples_linspace_probDist"
STD_BASE_DIR = "experiments_data_samples_linspace_std"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_std_linspace")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_std_linspace", "std_generation_master.log")

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

def chunk_deviations(devs, num_processes):
    """Split deviation list into chunks for processes"""
    chunk_size = len(devs) // num_processes
    remainder = len(devs) % num_processes
    
    chunks = []
    start_idx = 0
    
    for i in range(num_processes):
        # Add one extra deviation to the first 'remainder' processes
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        if start_idx < len(devs):
            chunks.append(devs[start_idx:end_idx])
        else:
            chunks.append([])  # Empty chunk if we have more processes than deviations
        
        start_idx = end_idx
    
    return chunks

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

def setup_process_logging(process_id, dev_chunk, theta=None):
    """Setup logging for individual processes handling multiple deviations"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format deviation range for filename
    if len(dev_chunk) > 0:
        first_dev = dev_chunk[0][1] if isinstance(dev_chunk[0], tuple) else dev_chunk[0]
        last_dev = dev_chunk[-1][1] if isinstance(dev_chunk[-1], tuple) else dev_chunk[-1]
        dev_range_str = f"dev{first_dev:.3f}to{last_dev:.3f}"
    else:
        dev_range_str = "empty"
    
    # Format theta for filename
    if theta is not None:
        theta_str = f"_theta{theta:.6f}"
    else:
        theta_str = ""
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process{process_id}_{dev_range_str}{theta_str}_std.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"process_{process_id}_std")
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
    file_formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [PROC:%(name)s] %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('[PROC:%(name)s] %(message)s')
    
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
# UTILITY FUNCTIONS
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

def generate_std_for_dev_chunk(chunk_args):
    """
    Worker function to generate standard deviation data for a chunk of deviation values.
    
    Args:
        chunk_args: Tuple containing (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, probdist_base_dir, std_base_dir)
    
    Returns:
        dict: Results from the std generation process
    """
    dev_chunk, process_id, N, steps, samples_count, theta_param, shutdown_flag, probdist_base_dir, std_base_dir = chunk_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(process_id, dev_chunk, theta_param)
    
    try:
        logger.info(f"=== STANDARD DEVIATION GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Process ID: {process_id}")
        logger.info(f"Deviation chunk size: {len(dev_chunk)}")
        if len(dev_chunk) > 0:
            first_dev = dev_chunk[0][1] if isinstance(dev_chunk[0], tuple) else dev_chunk[0]
            last_dev = dev_chunk[-1][1] if isinstance(dev_chunk[-1], tuple) else dev_chunk[-1]
            logger.info(f"Deviation range: {first_dev:.4f} to {last_dev:.4f}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        
        from smart_loading_static import format_theta_for_directory
        
        chunk_start_time = time.time()
        chunk_generated_count = 0
        chunk_skipped_count = 0
        processed_devs = 0
        
        # Process each deviation in the chunk
        for dev in dev_chunk:
            dev_start_time = time.time()
            logger.info(f"Processing deviation {processed_devs+1}/{len(dev_chunk)}: {dev}")
            
            # Log initial system resources
            log_system_resources(logger, "[WORKER]")
            
            # Format deviation for directory name
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_min, dev_max = dev
                dev_folder = f"dev_min{dev_min:.3f}_max{dev_max:.3f}"
            else:
                dev_folder = f"dev_{dev:.3f}"
            
            # Create directory structure for probdist and std
            theta_folder = format_theta_for_directory(theta_param)
            
            # ProbDist source directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            probdist_exp_dir = os.path.join(probdist_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            # Std target directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            std_exp_dir = os.path.join(std_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            logger.info(f"ProbDist source: {probdist_exp_dir}")
            logger.info(f"Std target: {std_exp_dir}")
            
            # Check if probDist directory exists
            if not os.path.exists(probdist_exp_dir):
                logger.error(f"ProbDist directory not found: {probdist_exp_dir}")
                continue
            
            # Check if std file already exists and is valid
            os.makedirs(std_exp_dir, exist_ok=True)
            std_file = os.path.join(std_exp_dir, "std_vs_time.pkl")
            
            skipped = False
            if validate_std_file(std_file):
                logger.info(f"Valid std file already exists: {std_file}")
                skipped = True
                chunk_skipped_count += 1
            else:
                # Load probability distributions
                prob_distributions = load_probability_distributions_for_dev(probdist_exp_dir, N, steps, logger)
                if not prob_distributions or len(prob_distributions) == 0:
                    logger.error(f"No probability distributions loaded")
                    continue
                
                # Calculate standard deviations
                logger.info(f"Calculating standard deviations for {len(prob_distributions)} time steps...")
                domain = np.arange(N) - N//2  # Center domain around 0
                std_values = prob_distributions2std(prob_distributions, domain)
                valid_std_count = sum(1 for s in std_values if s is not None and s > 0)
                logger.info(f"Calculated {valid_std_count}/{len(std_values)} valid standard deviations")
                
                if valid_std_count == 0:
                    logger.error(f"No valid standard deviations calculated")
                    continue
                
                # Save standard deviation data
                with open(std_file, 'wb') as f:
                    pickle.dump(std_values, f)
                logger.info(f"Standard deviation data saved to: {std_file}")
                
                if valid_std_count > 0:
                    final_std = [s for s in std_values if s is not None and s > 0][-1]
                    logger.info(f"Final std value: {final_std:.6f}")
                
                chunk_generated_count += 1
            
            dev_time = time.time() - dev_start_time
            processed_devs += 1
            
            # Format dev for display
            if isinstance(dev, tuple):
                dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            
            action = "skipped" if skipped else "generated"
            logger.info(f"Deviation {dev_str} completed: {action} in {dev_time:.1f}s")
            
            # Force garbage collection to free memory
            gc.collect()
        
        chunk_time = time.time() - chunk_start_time
        
        logger.info(f"=== STD GENERATION COMPLETED ===")
        logger.info(f"Process {process_id}: {processed_devs} deviations processed")
        logger.info(f"Actions: {chunk_generated_count} generated, {chunk_skipped_count} skipped")
        logger.info(f"Total time: {chunk_time:.1f}s")
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": processed_devs,
            "generated_count": chunk_generated_count,
            "skipped_count": chunk_skipped_count,
            "total_time": chunk_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error in std generation for process {process_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": 0,
            "generated_count": 0,
            "skipped_count": 0,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for standard deviation generation."""
    
    print("=== STANDARD DEVIATION GENERATION - LINSPACE VERSION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    print(f"Total deviations: {len(devs)}")
    print(f"Multiprocessing: {NUM_PROCESSES} processes")
    print(f"ProbDist source: {PROBDIST_BASE_DIR}")
    print(f"Std target: {STD_BASE_DIR}")
    print("=" * 70)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== STANDARD DEVIATION GENERATION - LINSPACE VERSION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    master_logger.info(f"Total deviations: {len(devs)}")
    master_logger.info(f"Processes: {NUM_PROCESSES}")
    
    start_time = time.time()
    
    # Split deviations into chunks for processes
    dev_chunks = chunk_deviations(devs, NUM_PROCESSES)
    
    # Log chunk distribution
    master_logger.info("DEVIATION CHUNK DISTRIBUTION:")
    for i, chunk in enumerate(dev_chunks):
        if len(chunk) > 0:
            first_dev = chunk[0][1] if isinstance(chunk[0], tuple) else chunk[0]
            last_dev = chunk[-1][1] if isinstance(chunk[-1], tuple) else chunk[-1]
            master_logger.info(f"  Process {i}: {len(chunk)} deviations ({first_dev:.4f} to {last_dev:.4f})")
        else:
            master_logger.info(f"  Process {i}: 0 deviations (empty)")
    
    # Prepare a shared shutdown flag for workers
    manager = mp.Manager()
    shutdown_flag = manager.Value('b', False)
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev_chunk in enumerate(dev_chunks):
        if len(dev_chunk) > 0:  # Only create processes for non-empty chunks
            args = (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, PROBDIST_BASE_DIR, STD_BASE_DIR)
            process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for std generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            # Submit all processes
            future_to_process = {}
            for args in process_args:
                future = executor.submit(generate_std_for_dev_chunk, args)
                future_to_process[future] = args[1]  # process_id
            
            # Collect results with timeout handling
            for future in as_completed(future_to_process, timeout=PROCESS_TIMEOUT * len(process_args)):
                process_id = future_to_process[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    if result["success"]:
                        master_logger.info(f"Process {process_id} completed successfully: "
                                         f"{result['processed_devs']} deviations, "
                                         f"{result['generated_count']} generated, "
                                         f"{result['skipped_count']} skipped, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process {process_id} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process {process_id} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": "Timeout",
                        "processed_devs": 0, "generated_count": 0, "skipped_count": 0,
                        "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process {process_id} crashed: {str(e)}")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": str(e),
                        "processed_devs": 0, "generated_count": 0, "skipped_count": 0,
                        "total_time": 0
                    })
    
    except KeyboardInterrupt:
        master_logger.warning("Interrupted by user")
        print("\n[INTERRUPT] Gracefully shutting down processes...")
    except Exception as e:
        master_logger.error(f"Critical error in multiprocessing: {str(e)}")
        raise
    
    total_time = time.time() - start_time
    
    # Generate summary
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    total_processed_devs = sum(r.get("processed_devs", 0) for r in process_results)
    total_generated = sum(r.get("generated_count", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_count", 0) for r in process_results)
    
    # === ARCHIVE CREATION ===
    if CREATE_TAR:
        import tarfile
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        main_tar_name = f"experiments_data_std_linspace_N{N}_samples{samples}_theta{theta:.6f}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        
        print(f"Creating archive: {main_tar_path}")
        master_logger.info(f"Creating archive: {main_tar_path}")
        
        try:
            with tarfile.open(main_tar_path, "w") as tar:
                tar.add(STD_BASE_DIR, arcname=os.path.basename(STD_BASE_DIR))
            print(f"Created main archive: {main_tar_path}")
            master_logger.info(f"Created main archive: {main_tar_path}")
        except Exception as e:
            master_logger.error(f"Failed to create archive: {e}")
    
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
                             f"Generated: {result.get('generated_count', 0)}, "
                             f"Skipped: {result.get('skipped_count', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Process {result['process_id']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "total_processed_devs": total_processed_devs,
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
