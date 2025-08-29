#!/usr/bin/env python3

"""
Generate Probability Distributions from Samples - Linspace Version

This script generates probability distribution (.pkl) files from existing sample data
created by the linspace sample generation script. It processes multiple deviation values
using configurable multiprocessing, checking for missing or invalid probability distribution
files and creating them from the corresponding sample files.

Key Features:
- Configurable number of processes (each handling multiple deviation values)
- Smart file validation (checks if probDist files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_probdist_from_samples_linspace.py

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
import tarfile

# ============================================================================
# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create per-chunk and main tar archives
ARCHIVE_DIR = "experiments_archive_linspace"

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 4000                 # System size (reduced from 20000)
steps = N//4             # Time steps
samples = 20             # Samples per deviation
theta = math.pi/3        # Theta parameter for static noise

# Note: Sample generation often includes initial step (step 0) + evolution steps
# So actual sample data may have steps + 1 directories (0 to steps inclusive)
EXPECT_INITIAL_STEP = True  # Set to True if samples include step_0 as initial state


# Deviation values - LINSPACE BETWEEN 0.6 AND 1.0 WITH 100 VALUES
DEV_MIN = 0.6
DEV_MAX = 1.0
DEV_COUNT = 100
devs = [(0, dev) for dev in np.linspace(DEV_MIN, DEV_MAX, DEV_COUNT)]

# Multiprocessing configuration
NUM_PROCESSES = 10        # Number of processes to use (CONFIGURABLE)

# Timeout configuration - Scale with problem size
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {NUM_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples_linspace"
PROBDIST_BASE_DIR = "experiments_data_samples_linspace_probDist"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_probdist_linspace")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_probdist_linspace", "probdist_generation_master.log")

# Global shutdown flag
SHUTDOWN_REQUESTED = False

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process{process_id}_{dev_range_str}{theta_str}_probdist.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"process_{process_id}_probdist")
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
    file_handler = logging.FileHandler(MASTER_LOG_FILE, mode='w')
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

def amp2prob(state):
    """Convert amplitude to probability distribution"""
    return np.abs(state)**2

def validate_probdist_file(file_path):
    """Check if a probability distribution file exists and is valid"""
    if not os.path.exists(file_path):
        return False
    
    try:
        # Check if file can be loaded and has reasonable content
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        # Basic validation: should be a numpy array
        if not isinstance(data, np.ndarray):
            return False
        
        # Should have reasonable size
        if len(data) == 0:
            return False
        
        # Values should be non-negative (probabilities)
        if np.any(data < 0):
            return False
        
        return True
    except Exception:
        return False

# ============================================================================
# PROBABILITY DISTRIBUTION GENERATION FUNCTIONS
# ============================================================================

def generate_step_probdist(samples_dir, target_dir, step_idx, N, samples_count, logger):
    """
    Generate probability distribution for a single step from sample files.
    Uses incremental mean calculation to be memory efficient.
    """
    try:
        step_dir = os.path.join(samples_dir, f"step_{step_idx}")
        
        if not os.path.exists(step_dir):
            logger.warning(f"Step directory not found: {step_dir}")
            return False
        
        mean_prob_dist = None
        valid_samples = 0
        
        for sample_idx in range(samples_count):
            # Use the actual filename format found in the directories
            filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
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
            
            # Only log completion for debugging
            if step_idx % 100 == 0 or step_idx < 10:
                logger.info(f"Generated probDist for step {step_idx} from {valid_samples} samples")
            return True
        else:
            logger.warning(f"No valid samples found for step {step_idx}")
            return False
        
    except Exception as e:
        logger.error(f"Error generating probDist for step {step_idx}: {str(e)}")
        return False

def generate_probdist_for_dev_chunk(chunk_args):
    """
    Worker function to generate probability distributions for a chunk of deviation values.
    
    Args:
        chunk_args: Tuple containing (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, samples_base_dir, probdist_base_dir)
    
    Returns:
        dict: Results from the probdist generation process
    """
    dev_chunk, process_id, N, steps, samples_count, theta_param, shutdown_flag, samples_base_dir, probdist_base_dir = chunk_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(process_id, dev_chunk, theta_param)
    
    try:
        logger.info(f"=== PROBDIST GENERATION STARTED ===")
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
        chunk_computed_steps = 0
        chunk_skipped_steps = 0
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
            
            # Create directory structure for samples and probdist
            theta_folder = format_theta_for_directory(theta_param)
            
            # Samples source directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            samples_exp_dir = os.path.join(samples_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            # ProbDist target directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            probdist_exp_dir = os.path.join(probdist_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            logger.info(f"Samples source: {samples_exp_dir}")
            logger.info(f"ProbDist target: {probdist_exp_dir}")
            
            # Check if samples directory exists
            if not os.path.exists(samples_exp_dir):
                logger.error(f"Samples directory not found: {samples_exp_dir}")
                continue
            
            # Determine the actual number of steps to process from the samples directory
            step_dirs = [d for d in os.listdir(samples_exp_dir) if os.path.isdir(os.path.join(samples_exp_dir, d)) and d.startswith("step_")]
            actual_steps = len(step_dirs)
            
            # Use the actual number of steps found in the samples directory
            steps_to_process = actual_steps
            logger.info(f"Found {actual_steps} step directories in samples (expected {steps} or {steps + 1})")
            logger.info(f"Will process {steps_to_process} steps for probDist generation")
            
            dev_computed_steps = 0
            dev_skipped_steps = 0
            
            # Process each step
            for step_idx in range(steps_to_process):
                # Check if probDist file exists and is valid
                probdist_file = os.path.join(probdist_exp_dir, f"mean_step_{step_idx}.pkl")
                
                if validate_probdist_file(probdist_file):
                    dev_skipped_steps += 1
                    if step_idx % 100 == 0:
                        logger.info(f"    Step {step_idx+1}/{steps_to_process} already exists, skipping")
                else:
                    # Only log generation for debugging
                    if step_idx % 100 == 0:
                        logger.info(f"    Step {step_idx+1}/{steps_to_process} processing...")
                        
                    if generate_step_probdist(samples_exp_dir, probdist_exp_dir, step_idx, N, samples_count, logger):
                        dev_computed_steps += 1
                    else:
                        logger.error(f"Failed to generate probDist for step {step_idx}")
                
                # Force garbage collection periodically to keep memory usage low
                if step_idx % 50 == 0:
                    gc.collect()
            
            dev_time = time.time() - dev_start_time
            processed_devs += 1
            chunk_computed_steps += dev_computed_steps
            chunk_skipped_steps += dev_skipped_steps
            
            # Format dev for display
            if isinstance(dev, tuple):
                dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            
            logger.info(f"Deviation {dev_str} completed: {dev_computed_steps} computed, {dev_skipped_steps} skipped in {dev_time:.1f}s")
        
        chunk_time = time.time() - chunk_start_time
        
        logger.info(f"=== PROBDIST GENERATION COMPLETED ===")
        logger.info(f"Process {process_id}: {processed_devs} deviations processed")
        logger.info(f"Total steps: {chunk_computed_steps} computed, {chunk_skipped_steps} skipped")
        logger.info(f"Total time: {chunk_time:.1f}s")
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": processed_devs,
            "computed_steps": chunk_computed_steps,
            "skipped_steps": chunk_skipped_steps,
            "total_steps": steps * len(dev_chunk),
            "total_time": chunk_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error in probdist generation for process {process_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": 0,
            "computed_steps": 0,
            "skipped_steps": 0,
            "total_steps": steps * len(dev_chunk),
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for probability distribution generation."""
    
    print("=== PROBABILITY DISTRIBUTION GENERATION - LINSPACE VERSION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    print(f"Total deviations: {len(devs)}")
    print(f"Multiprocessing: {NUM_PROCESSES} processes")
    print(f"Samples source: {SAMPLES_BASE_DIR}")
    print(f"ProbDist target: {PROBDIST_BASE_DIR}")
    print("=" * 70)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== PROBABILITY DISTRIBUTION GENERATION - LINSPACE VERSION STARTED ===")
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
            args = (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, SAMPLES_BASE_DIR, PROBDIST_BASE_DIR)
            process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for probDist generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            # Submit all processes
            future_to_process = {}
            for args in process_args:
                future = executor.submit(generate_probdist_for_dev_chunk, args)
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
                                         f"{result['computed_steps']} computed, "
                                         f"{result['skipped_steps']} skipped, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process {process_id} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process {process_id} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": "Timeout",
                        "processed_devs": 0, "computed_steps": 0, "skipped_steps": 0,
                        "total_steps": 0, "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process {process_id} crashed: {str(e)}")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": str(e),
                        "processed_devs": 0, "computed_steps": 0, "skipped_steps": 0,
                        "total_steps": 0, "total_time": 0
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
    total_computed = sum(r.get("computed_steps", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_steps", 0) for r in process_results)
    
    # === MAIN ARCHIVE CREATION ===
    if CREATE_TAR:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        main_tar_name = f"experiments_data_probDist_linspace_N{N}_samples{samples}_theta{theta:.6f}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        
        print(f"Creating archive: {main_tar_path}")
        master_logger.info(f"Creating archive: {main_tar_path}")
        
        try:
            with tarfile.open(main_tar_path, "w") as tar:
                tar.add(PROBDIST_BASE_DIR, arcname=os.path.basename(PROBDIST_BASE_DIR))
            print(f"Created main archive: {main_tar_path}")
            master_logger.info(f"Created main archive: {main_tar_path}")
        except Exception as e:
            master_logger.error(f"Failed to create archive: {e}")
    
    # Calculate total execution time
    total_time = time.time() - start_time
    
    # Generate final summary with enhanced validation
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    total_processed_devs = sum(r.get("processed_devs", 0) for r in process_results)
    total_computed = sum(r.get("computed_steps", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_steps", 0) for r in process_results)
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    print(f"Steps: {total_computed} computed, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    master_logger.info(f"Steps: {total_computed} computed, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Process {result['process_id']}: SUCCESS - "
                             f"Devs: {result.get('processed_devs', 0)}, "
                             f"Computed: {result.get('computed_steps', 0)}, "
                             f"Skipped: {result.get('skipped_steps', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Process {result['process_id']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "total_processed_devs": total_processed_devs,
        "total_computed": total_computed,
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
