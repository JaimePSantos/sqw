#!/usr/bin/env python3

"""
Generate Survival Probability Data from Probability Distributions

This script generates survival probability data from existing probability distribution files.
It processes multiple deviation values in parallel, calculating survival probabilities for
configurable node ranges and saving the results for later plotting.

Key Features:
- Multi-process execution (one process per deviation value)
- Configurable survival probability ranges (centered and full system)
- Smart file validation (checks if survival files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_survival_probability.py

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
import signal
import math
import numpy as np

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 300                # System size
steps = N//4           # Time steps
samples = 5            # Samples per deviation
theta = math.pi/3      # Theta parameter for static noise

# Deviation values - TEST SET (matching other scripts)
devs = [
    (0,0),              # No noise
    (0, 0.1),           # Small noise range
    (0, 0.9),           # Medium noise range  
]

# Survival probability ranges configuration
SURVIVAL_RANGES = {
    "center": {
        "description": "Central nodes (N//4 around center)",
        "range_func": lambda N: range(N//2 - N//8, N//2 + N//8 + 1)
    },
    "system": {
        "description": "Full system",
        "range_func": lambda N: range(N)
    }
}

# Directory configuration
PROBDIST_BASE_DIR = "experiments_data_samples_probDist"
SURVIVAL_BASE_DIR = "experiments_data_samples_survival"
PROCESS_LOG_DIR = "generate_survival"

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())
PROCESS_TIMEOUT = 3600  # 1 hour timeout per process

# Logging configuration
MASTER_LOG_FILE = "generate_survival/survival_generation_master.log"

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_survival.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_survival")
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
    
    Structure: base_dir/tesselation_func_noise_type/theta_value/dev_range/N_value/samples_count
    Example: experiments_data_samples_survival/dummy_tesselation_func_static_noise/theta_1.047198/dev_min0.000_max0.000/N_20000/samples_40
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

def find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data", theta=None):
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

def resolve_node_position(node_spec, N):
    """
    Resolve node position specification to actual node index.
    
    Args:
        node_spec: Can be int, float, or string
                  - int: Direct node index
                  - float: Fraction of N (e.g., 0.5 for center)
                  - str: Special keywords ("center", "left", "right")
        N: System size
    
    Returns:
        int: Resolved node index
    """
    if isinstance(node_spec, int):
        return node_spec
    elif isinstance(node_spec, float):
        return int(node_spec * N)
    elif isinstance(node_spec, str):
        if node_spec.lower() == "center":
            return N // 2
        elif node_spec.lower() == "left":
            return 0
        elif node_spec.lower() == "right":
            return N - 1
        else:
            raise ValueError(f"Unknown node specification: {node_spec}")
    else:
        raise ValueError(f"Invalid node specification type: {type(node_spec)}")

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

def generate_survival_for_dev(dev_args):
    """
    Worker function to generate survival probability data for a single deviation value.
    
    Args:
        dev_args: Tuple containing (dev, process_id, N, steps, samples, theta)
    
    Returns:
        dict: Results from the survival probability generation process
    """
    dev, process_id, N, steps, samples_count, theta_param = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"=== SURVIVAL PROBABILITY GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        logger.info(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
        
        dev_start_time = time.time()
        
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
        
        survival_exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=SURVIVAL_BASE_DIR, theta=theta_param, samples=samples_count
        )
        
        logger.info(f"ProbDist source: {probdist_exp_dir}")
        logger.info(f"Survival target: {survival_exp_dir}")
        logger.info(f"Source format: {probdist_format}")
        
        # Check if probDist directory exists
        if not os.path.exists(probdist_exp_dir):
            logger.error(f"ProbDist directory not found: {probdist_exp_dir}")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": f"ProbDist directory not found: {probdist_exp_dir}",
                "log_file": log_file, "total_time": 0
            }
        
        # Check if survival file already exists and is valid
        os.makedirs(survival_exp_dir, exist_ok=True)
        survival_file = os.path.join(survival_exp_dir, "survival_vs_time.pkl")
        
        if validate_survival_file(survival_file):
            logger.info(f"Valid survival file already exists: {survival_file}")
            return {
                "dev": dev, "process_id": process_id, "success": True,
                "error": None, "log_file": log_file, "total_time": 0,
                "action": "skipped", "message": "Valid survival file already exists"
            }
        
        # Load probability distributions
        prob_distributions = load_probability_distributions_for_dev(probdist_exp_dir, N, steps, logger)
        
        if not prob_distributions or len(prob_distributions) == 0:
            logger.error(f"No probability distributions loaded")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": "No probability distributions loaded",
                "log_file": log_file, "total_time": 0
            }
        
        # Generate survival probability data
        logger.info(f"Generating survival probability data for {len(prob_distributions)} time steps...")
        survival_data = generate_survival_probability_data(prob_distributions, N, logger)
        
        # Validate results
        total_ranges = len(SURVIVAL_RANGES)
        valid_ranges = sum(1 for range_name, range_data in survival_data.items() 
                          if any(p is not None for p in range_data))
        
        if valid_ranges == 0:
            logger.error(f"No valid survival probabilities calculated")
            return {
                "dev": dev, "process_id": process_id, "success": False,
                "error": "No valid survival probabilities calculated",
                "log_file": log_file, "total_time": 0
            }
        
        # Save survival probability data
        with open(survival_file, 'wb') as f:
            pickle.dump(survival_data, f)
        
        logger.info(f"Survival probability data saved to: {survival_file}")
        
        dev_time = time.time() - dev_start_time
        
        logger.info(f"=== SURVIVAL PROBABILITY GENERATION COMPLETED ===")
        logger.info(f"Valid ranges: {valid_ranges}/{total_ranges}")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "error": None,
            "valid_ranges": valid_ranges,
            "total_ranges": total_ranges,
            "log_file": log_file,
            "total_time": dev_time,
            "action": "generated",
            "message": f"Generated survival data for {valid_ranges} ranges"
        }
        
    except Exception as e:
        error_msg = f"Error in survival generation for dev {dev_str}: {str(e)}"
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
    """Main execution function for survival probability generation."""
    
    print("=== SURVIVAL PROBABILITY GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"ProbDist source: {PROBDIST_BASE_DIR}")
    print(f"Survival target: {SURVIVAL_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== SURVIVAL PROBABILITY GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Survival ranges: {list(SURVIVAL_RANGES.keys())}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, theta)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for survival generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_survival_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    if result["success"]:
                        action = result.get("action", "processed")
                        message = result.get("message", "")
                        master_logger.info(f"Process for dev {dev} completed successfully: "
                                         f"{action} - {message}, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process for dev {dev} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process for dev {dev} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "dev": dev, "success": False, "error": "Timeout",
                        "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process for dev {dev} crashed: {str(e)}")
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
