#!/usr/bin/env python3

"""
Recreate Probability Distributions from Samples

This script recreates probability distribution (.pkl) files from existing sample data.
It processes multiple deviation values in parallel, checking for missing or invalid
probability distribution files and recreating them from the corresponding sample files.

Key Features:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if probDist files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python recreate_probdist_from_samples.py

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

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 20000                # System size (small for testing)
steps = N//4           # Time steps (25 for N=100)
samples = 40           # Samples per deviation (small for testing)
theta = math.pi/3      # Theta parameter for static noise

# Deviation values - TEST SET (matching generate_samples.py)
devs = [
    # (0,0),              # No noise
    # (0, 0.2),           # Small noise range
    # (0, 0.5),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
]

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples"
PROBDIST_BASE_DIR = "experiments_data_samples_probDist"
PROCESS_LOG_DIR = "recreate_probDist"

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())
PROCESS_TIMEOUT = 7200  # 2 hour timeout per process

# Logging configuration
MASTER_LOG_FILE = "recreate_probDist/probdist_recreation_master.log"

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
    file_handler = logging.FileHandler(log_filename, mode='w')
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
# DIRECTORY AND FILE MANAGEMENT
# ============================================================================

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data", theta=None, samples=None):
    """
    Get experiment directory path with proper structure.
    Handles both old and new directory formats.
    """
    # Handle deviation format
    if noise_params and len(noise_params) > 0:
        dev = noise_params[0]
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"{float(dev):.3f}"
    else:
        dev_str = "0.000"
    
    # Format theta for directory
    if theta is not None:
        theta_str = f"theta_{theta:.6f}"
    else:
        theta_str = "theta_default"
    
    # Build directory path
    if noise_type == "static_noise":
        noise_dir = f"static_dev_{dev_str}"
        if samples is not None:
            noise_dir += f"_samples_{samples}"
        
        exp_dir = os.path.join(base_dir, theta_str, f"N_{N}", noise_dir)
    else:
        exp_dir = os.path.join(base_dir, f"N_{N}", f"dev_{dev_str}")
    
    return exp_dir

def find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data", theta=None):
    """
    Find experiment directory with flexible format matching.
    Tries different directory structures to find existing data.
    """
    # Try new format first (with samples)
    for samples_try in [samples, None]:  # Try with and without samples in path
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta, samples_try)
        if os.path.exists(exp_dir):
            return exp_dir, "new_format"
    
    # Try old format variations
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
    return get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta, samples), "new_format"

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
# PROBABILITY DISTRIBUTION RECREATION FUNCTIONS
# ============================================================================

def recreate_step_probdist(samples_dir, target_dir, step_idx, N, samples_count, logger):
    """
    Recreate probability distribution for a specific step from sample files.
    
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
            logger.warning(f"Step directory not found: {step_dir}")
            return False
        
        # Collect all valid samples for this step
        valid_prob_dists = []
        
        for sample_idx in range(samples_count):
            sample_file = os.path.join(step_dir, f"sample_{sample_idx}.pkl")
            sample_data = load_sample_file(sample_file)
            
            if sample_data is not None:
                # Convert amplitude to probability
                prob_dist = amp2prob(sample_data)
                valid_prob_dists.append(prob_dist)
            else:
                logger.warning(f"Failed to load sample {sample_idx} for step {step_idx}")
        
        if not valid_prob_dists:
            logger.error(f"No valid samples found for step {step_idx}")
            return False
        
        # Calculate mean probability distribution
        mean_prob_dist = np.mean(valid_prob_dists, axis=0)
        
        # Save mean probability distribution
        os.makedirs(target_dir, exist_ok=True)
        mean_filepath = os.path.join(target_dir, f"mean_step_{step_idx}.pkl")
        
        with open(mean_filepath, 'wb') as f:
            pickle.dump(mean_prob_dist, f)
        
        logger.info(f"Recreated probDist for step {step_idx} from {len(valid_prob_dists)} samples")
        return True
        
    except Exception as e:
        logger.error(f"Error recreating probDist for step {step_idx}: {str(e)}")
        return False

def recreate_probdist_for_dev(dev_args):
    """
    Worker function to recreate probability distributions for a single deviation value.
    
    Args:
        dev_args: Tuple containing (dev, process_id, N, steps, samples, theta)
    
    Returns:
        dict: Results from the recreation process
    """
    dev, process_id, N, steps, samples_count, theta_param = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"=== PROBDIST RECREATION STARTED ===")
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
        
        # Get source and target directories
        noise_params = [dev]
        samples_exp_dir, found_format = find_experiment_dir_flexible(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=SAMPLES_BASE_DIR, theta=theta_param
        )
        
        probdist_exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=PROBDIST_BASE_DIR, theta=theta_param, samples=samples_count
        )
        
        logger.info(f"Samples source: {samples_exp_dir}")
        logger.info(f"ProbDist target: {probdist_exp_dir}")
        logger.info(f"Source format: {found_format}")
        
        # Check if samples directory exists
        if not os.path.exists(samples_exp_dir):
            logger.error(f"Samples directory not found: {samples_exp_dir}")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": f"Samples directory not found: {samples_exp_dir}",
                "processed_steps": 0, "total_steps": steps,
                "log_file": log_file, "total_time": 0
            }
        
        # Process each step
        processed_steps = 0
        skipped_steps = 0
        recreated_steps = 0
        
        for step_idx in range(steps):
            global SHUTDOWN_REQUESTED
            if SHUTDOWN_REQUESTED:
                logger.warning(f"Shutdown requested, stopping at step {step_idx}")
                break
            
            # Check if probDist file exists and is valid
            probdist_file = os.path.join(probdist_exp_dir, f"mean_step_{step_idx}.pkl")
            
            if validate_probdist_file(probdist_file):
                logger.debug(f"Valid probDist exists for step {step_idx}, skipping")
                skipped_steps += 1
            else:
                logger.info(f"Recreating probDist for step {step_idx}...")
                if recreate_step_probdist(samples_exp_dir, probdist_exp_dir, step_idx, N, samples_count, logger):
                    recreated_steps += 1
                else:
                    logger.error(f"Failed to recreate probDist for step {step_idx}")
            
            processed_steps += 1
            
            # Log progress every 100 steps
            if step_idx % 100 == 0 and step_idx > 0:
                elapsed = time.time() - dev_start_time
                eta = (elapsed / step_idx) * (steps - step_idx)
                logger.info(f"Progress: {step_idx}/{steps} steps - Elapsed: {elapsed/60:.1f}m - ETA: {eta/60:.1f}m")
        
        dev_time = time.time() - dev_start_time
        
        logger.info(f"=== PROBDIST RECREATION COMPLETED ===")
        logger.info(f"Processed: {processed_steps}/{steps} steps")
        logger.info(f"Skipped (already valid): {skipped_steps}")
        logger.info(f"Recreated: {recreated_steps}")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "error": None,
            "processed_steps": processed_steps,
            "total_steps": steps,
            "skipped_steps": skipped_steps,
            "recreated_steps": recreated_steps,
            "log_file": log_file,
            "total_time": dev_time
        }
        
    except Exception as e:
        error_msg = f"Error in probDist recreation for dev {dev_str}: {str(e)}"
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
            "recreated_steps": 0,
            "log_file": log_file,
            "total_time": 0
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for probability distribution recreation."""
    
    print("=== PROBABILITY DISTRIBUTION RECREATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"Samples source: {SAMPLES_BASE_DIR}")
    print(f"ProbDist target: {PROBDIST_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== PROBABILITY DISTRIBUTION RECREATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, theta)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for probDist recreation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(recreate_probdist_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    if result["success"]:
                        master_logger.info(f"Process for dev {dev} completed successfully: "
                                         f"{result['recreated_steps']} recreated, "
                                         f"{result['skipped_steps']} skipped, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process for dev {dev} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process for dev {dev} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "dev": dev, "success": False, "error": "Timeout",
                        "processed_steps": 0, "total_steps": steps,
                        "skipped_steps": 0, "recreated_steps": 0,
                        "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process for dev {dev} crashed: {str(e)}")
                    process_results.append({
                        "dev": dev, "success": False, "error": str(e),
                        "processed_steps": 0, "total_steps": steps,
                        "skipped_steps": 0, "recreated_steps": 0,
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
    total_recreated = sum(r.get("recreated_steps", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_steps", 0) for r in process_results)
    total_processed = sum(r.get("processed_steps", 0) for r in process_results)
    
    print(f"\n=== RECREATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Steps: {total_processed} processed, {total_recreated} recreated, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== RECREATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Steps: {total_processed} processed, {total_recreated} recreated, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Dev {result['dev']}: SUCCESS - "
                             f"Recreated: {result.get('recreated_steps', 0)}, "
                             f"Skipped: {result.get('skipped_steps', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Dev {result['dev']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
        "total_recreated": total_recreated,
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
