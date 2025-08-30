#!/usr/bin/env python3

"""
Generate Probability Distributions from Dynamic Samples

This script generates probability distribution (.pkl) files from existing dynamic sample data.
It processes multiple deviation values in parallel, checking for missing or invalid
probability distribution files and creating them from the corresponding sample files.

Key Features:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if probDist files exist and have valid data)
- Dynamic noise experiment directory structure handling
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_dynamic_probdist_from_samples.py

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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 100                # System size (small for testing dynamic noise)
steps = N//4           # Time steps
samples = 5            # Samples per deviation (reduced for testing)
base_theta = math.pi/3 # Base theta parameter for dynamic angle noise

# Deviation values - Dynamic noise format (angle deviations)
devs = [
    0,                  # No noise
    (math.pi/3)/4,      # Small noise
    (math.pi/3)/2,      # Medium noise  
]

# Note: Dynamic sample generation includes initial step (step 0) + evolution steps
# So actual sample data has steps + 1 directories (0 to steps inclusive)
EXPECT_INITIAL_STEP = True  # Set to True if samples include step_0 as initial state

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples_dynamic"
PROBDIST_BASE_DIR = "experiments_data_samples_dynamic_probDist"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_dynamic_probdist")

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
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_dynamic_probdist", "dynamic_probdist_generation_master.log")

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}{theta_str}_dynamic_probdist.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_dynamic_probdist")
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

def find_dynamic_samples_directory_for_config(base_dir, N, base_theta, dev, samples_count, logger):
    """
    Find the dynamic samples directory that contains data for the specified configuration.
    
    Returns:
        tuple: (samples_dir_path, format_type) or (None, None) if not found
    """
    logger.info(f"Searching for dynamic samples directory: N={N}, samples={samples_count}, dev={dev}, base_theta={base_theta:.6f}")
    
    # Dynamic format structure
    has_noise = dev > 0
    noise_params = [dev]
    
    samples_path = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=base_dir, 
        base_theta=base_theta
    )
    
    if os.path.exists(samples_path):
        # Check if we have the expected number of step directories
        step_dirs = [d for d in os.listdir(samples_path) if os.path.isdir(os.path.join(samples_path, d)) and d.startswith("step_")]
        found_steps = len(step_dirs)
        
        # Check sample count in first step directory
        if step_dirs:
            first_step_dir = os.path.join(samples_path, step_dirs[0])
            sample_files = [f for f in os.listdir(first_step_dir) if f.startswith("final_step_") and f.endswith(".pkl")]
            found_samples = len(sample_files)
            
            logger.info(f"Found dynamic samples directory: {samples_path}")
            logger.info(f"  Steps: {found_steps}, Samples: {found_samples}")
            
            return samples_path, "dynamic_format"
    
    logger.warning(f"No valid dynamic samples directory found for configuration: N={N}, samples={samples_count}, dev={dev}")
    return None, None

# ============================================================================
# DIRECTORY AND FILE MANAGEMENT
# ============================================================================

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
            # Check if data is a valid numpy array or list with reasonable content
            if data is None:
                return False
            return True
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
            return pickle.load(f)
    except (pickle.PickleError, EOFError, ValueError, TypeError):
        return None

def validate_dynamic_samples_configuration(samples_exp_dir, expected_samples, expected_steps, expected_N, expected_base_theta, logger):
    """
    Validate that the dynamic samples directory contains data for the expected configuration.
    Checks both the directory structure and a sample of files to ensure consistency.
    
    Note: Accepts either expected_steps or expected_steps + 1 to handle cases where
    sample generation includes an initial step (step_0) plus evolution steps.
    
    Returns:
        dict: {"valid": bool, "found_samples": int, "found_steps": int, "issues": [str]}
    """
    issues = []
    found_steps = 0
    found_samples = 0
    
    logger.info(f"Validating dynamic samples configuration:")
    logger.info(f"  Expected: N={expected_N}, steps={expected_steps} (or {expected_steps + 1}), samples={expected_samples}, base_theta={expected_base_theta:.6f}")
    logger.info(f"  Directory: {samples_exp_dir}")
    
    if not os.path.exists(samples_exp_dir):
        issues.append(f"Samples directory does not exist: {samples_exp_dir}")
        return {"valid": False, "found_samples": 0, "found_steps": 0, "issues": issues}
    
    # Count available steps
    step_dirs = [d for d in os.listdir(samples_exp_dir) if os.path.isdir(os.path.join(samples_exp_dir, d)) and d.startswith("step_")]
    found_steps = len(step_dirs)
    
    if found_steps == 0:
        issues.append("No step directories found")
        return {"valid": False, "found_samples": 0, "found_steps": 0, "issues": issues}
    
    # Check a few steps to determine sample count
    sample_counts = []
    for step_dir_name in sorted(step_dirs)[:min(5, len(step_dirs))]:
        step_dir = os.path.join(samples_exp_dir, step_dir_name)
        sample_files = [f for f in os.listdir(step_dir) if f.startswith("final_step_") and f.endswith(".pkl")]
        sample_counts.append(len(sample_files))
    
    # Determine the actual sample count
    if sample_counts:
        found_samples = max(sample_counts)  # Use the maximum found
        if len(set(sample_counts)) > 1:
            issues.append(f"Inconsistent sample counts across steps: {set(sample_counts)}")
    
    # Validate against expected values - allow for steps or steps + 1
    if found_steps != expected_steps and found_steps != (expected_steps + 1):
        issues.append(f"Step count mismatch: expected {expected_steps} (or {expected_steps + 1}), found {found_steps}")
    
    if found_samples != expected_samples:
        issues.append(f"Sample count mismatch: expected {expected_samples}, found {found_samples}")
    
    # Log results
    if issues:
        logger.warning(f"Validation issues found: {issues}")
    else:
        logger.info(f"Validation passed: {found_steps} steps, {found_samples} samples")
    
    return {
        "valid": len(issues) == 0,
        "found_samples": found_samples,
        "found_steps": found_steps,
        "issues": issues
    }

# ============================================================================
# PROBABILITY DISTRIBUTION GENERATION FUNCTIONS
# ============================================================================

def generate_step_probdist(samples_dir, target_dir, step_idx, N, samples_count, logger):
    """
    Generate probability distribution for a specific step from sample files.
    Uses optimized streaming processing with incremental mean calculation to minimize memory usage.
    
    Args:
        samples_dir: Directory containing sample files
        target_dir: Directory to save probability distribution
        step_idx: Step index to process
        N: System size
        samples_count: Number of samples
        logger: Logger for this process
    
    Returns:
        tuple: (success: bool, was_skipped: bool) - (True, True) if skipped, (True, False) if computed, (False, False) if failed
    """
    try:
        import numpy as np
        
        # Create target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Check if output file already exists and is valid
        output_file = os.path.join(target_dir, f"mean_step_{step_idx}.pkl")
        if validate_probdist_file(output_file):
            logger.info(f"    Step {step_idx}: Already exists and valid, skipping")
            return True, True  # success=True, was_skipped=True
        
        # Find step directory
        step_dir = os.path.join(samples_dir, f"step_{step_idx}")
        if not os.path.exists(step_dir):
            logger.error(f"    Step {step_idx}: Step directory not found: {step_dir}")
            return False, False  # success=False, was_skipped=False
        
        logger.debug(f"    Step {step_idx}: Processing {samples_count} samples...")
        
        # Initialize running mean calculation
        running_mean = None
        valid_samples = 0
        
        # Process each sample file
        for sample_idx in range(samples_count):
            sample_file = os.path.join(step_dir, f"final_step_{step_idx}_sample{sample_idx}.pkl")
            
            # Load sample data
            sample_data = load_sample_file(sample_file)
            if sample_data is None:
                logger.warning(f"    Step {step_idx}: Failed to load sample {sample_idx}")
                continue
            
            # Convert to numpy array if needed
            if not isinstance(sample_data, np.ndarray):
                try:
                    sample_data = np.array(sample_data)
                except:
                    logger.warning(f"    Step {step_idx}: Could not convert sample {sample_idx} to numpy array")
                    continue
            
            # Get probability distribution (magnitude squared)
            prob_dist = np.abs(sample_data) ** 2
            
            # Update running mean
            if running_mean is None:
                running_mean = prob_dist.copy()
                valid_samples = 1
            else:
                # Incremental mean: new_mean = old_mean + (new_value - old_mean) / (count + 1)
                valid_samples += 1
                running_mean += (prob_dist - running_mean) / valid_samples
            
            # Clean up
            del sample_data, prob_dist
            gc.collect()
        
        if valid_samples == 0:
            logger.error(f"    Step {step_idx}: No valid samples found")
            return False, False  # success=False, was_skipped=False
        
        if valid_samples != samples_count:
            logger.warning(f"    Step {step_idx}: Only {valid_samples}/{samples_count} samples were valid")
        
        # Save the mean probability distribution
        with open(output_file, 'wb') as f:
            pickle.dump(running_mean, f)
        
        logger.debug(f"    Step {step_idx}: Generated probdist from {valid_samples} samples")
        
        # Clean up
        del running_mean
        gc.collect()
        
        return True, False  # success=True, was_skipped=False (actually computed)
        
    except Exception as e:
        logger.error(f"    Step {step_idx}: Error generating probdist: {e}")
        logger.error(traceback.format_exc())
        return False, False  # success=False, was_skipped=False

def generate_dynamic_probdist_for_dev(dev_args):
    """
    Worker function to generate probability distributions for a single deviation value.
    dev_args: (dev, process_id, N, steps, samples, base_theta, source_base_dir, target_base_dir)
    """
    dev, process_id, N, steps, samples_count, base_theta_param, source_base_dir, target_base_dir = dev_args
    
    # Setup logging for this process
    dev_rounded = round(dev, 6)
    dev_str = f"{dev_rounded:.6f}"
    logger, log_file = setup_process_logging(dev_str, process_id, base_theta_param)
    
    try:
        logger.info(f"=== DYNAMIC PROBDIST GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, base_theta={base_theta_param:.6f}")
        
        dev_start_time = time.time()
        
        # Find samples directory
        samples_dir, format_type = find_dynamic_samples_directory_for_config(
            source_base_dir, N, base_theta_param, dev, samples_count, logger
        )
        
        if samples_dir is None:
            error_msg = f"No valid samples directory found for dev {dev_str}"
            logger.error(error_msg)
            return {
                "dev": dev, "process_id": process_id, "success": False, "error": error_msg,
                "computed_steps": 0, "skipped_steps": 0, "total_steps": steps + 1,
                "total_time": 0, "log_file": log_file
            }
        
        # Validate samples configuration
        validation_result = validate_dynamic_samples_configuration(
            samples_dir, samples_count, steps, N, base_theta_param, logger
        )
        
        if not validation_result["valid"]:
            error_msg = f"Samples validation failed: {validation_result['issues']}"
            logger.error(error_msg)
            return {
                "dev": dev, "process_id": process_id, "success": False, "error": error_msg,
                "computed_steps": 0, "skipped_steps": 0, "total_steps": steps + 1,
                "total_time": 0, "log_file": log_file
            }
        
        # Determine target directory for probability distributions
        has_noise = dev > 0
        noise_params = [dev]
        
        target_dir = get_dynamic_experiment_dir(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, 
            base_dir=target_base_dir, 
            base_theta=base_theta_param
        )
        
        logger.info(f"Source directory: {samples_dir}")
        logger.info(f"Target directory: {target_dir}")
        logger.info(f"Processing {validation_result['found_steps']} steps with {validation_result['found_samples']} samples each")
        
        # Process each step
        actual_steps = validation_result['found_steps']
        computed_steps = 0
        skipped_steps = 0
        
        for step_idx in range(actual_steps):
            step_success, was_skipped = generate_step_probdist(samples_dir, target_dir, step_idx, N, samples_count, logger)
            
            if step_success:
                if was_skipped:
                    skipped_steps += 1
                else:
                    computed_steps += 1
                if step_idx % 5 == 0 or step_idx == actual_steps - 1:
                    logger.info(f"Progress: {step_idx + 1}/{actual_steps} steps processed")
            else:
                logger.warning(f"Failed to process step {step_idx}")
                # Count failures separately, don't add to computed or skipped
        
        dev_time = time.time() - dev_start_time
        
        logger.info(f"=== DYNAMIC PROBDIST GENERATION COMPLETED ===")
        logger.info(f"Deviation {dev_str}: {computed_steps} computed, {skipped_steps} skipped")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "error": None,
            "computed_steps": computed_steps,
            "skipped_steps": skipped_steps,
            "total_steps": actual_steps,
            "total_time": dev_time,
            "log_file": log_file
        }
        
    except Exception as e:
        dev_rounded = round(dev, 6)
        dev_str = f"{dev_rounded:.4f}"
        error_msg = f"Error in dynamic probdist generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": False,
            "error": error_msg,
            "computed_steps": 0,
            "skipped_steps": 0,
            "total_steps": steps + 1,
            "total_time": 0,
            "log_file": log_file
        }

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for dynamic probability distribution generation."""
    
    print("=== DYNAMIC PROBABILITY DISTRIBUTION GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"Samples source: {SAMPLES_BASE_DIR}")
    print(f"ProbDist target: {PROBDIST_BASE_DIR}")
    print("=" * 60)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== DYNAMIC PROBABILITY DISTRIBUTION GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, base_theta, SAMPLES_BASE_DIR, PROBDIST_BASE_DIR)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for dynamic probdist generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_dynamic_probdist_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    if result["success"]:
                        dev_rounded = round(dev, 6)
                        master_logger.info(f"Dev {dev_rounded:.4f}: SUCCESS - "
                                         f"Computed: {result['computed_steps']}, "
                                         f"Skipped: {result['skipped_steps']}, "
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
                        "computed_steps": 0, "skipped_steps": 0, "total_time": 0
                    })
                except Exception as e:
                    dev_rounded = round(dev, 6)
                    error_msg = f"Dev {dev_rounded:.4f}: EXCEPTION - {str(e)}"
                    master_logger.error(error_msg)
                    process_results.append({
                        "dev": dev, "success": False, "error": error_msg,
                        "computed_steps": 0, "skipped_steps": 0, "total_time": 0
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
    total_computed = sum(r.get("computed_steps", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_steps", 0) for r in process_results)
    
    print(f"\n=== DYNAMIC PROBDIST GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Steps: {total_computed} computed, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== DYNAMIC PROBDIST GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Steps: {total_computed} computed, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Dev {result['dev']}: SUCCESS - "
                             f"Computed: {result.get('computed_steps', 0)}, "
                             f"Skipped: {result.get('skipped_steps', 0)}, "
                             f"Time: {result.get('total_time', 0):.1f}s")
        else:
            master_logger.info(f"  Dev {result['dev']}: FAILED - {result.get('error', 'Unknown error')}")
    
    return {
        "total_time": total_time,
        "successful_processes": successful_processes,
        "failed_processes": failed_processes,
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
