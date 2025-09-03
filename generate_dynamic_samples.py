#!/usr/bin/env python3

"""
Generate Dynamic Sample Files for Quantum Walk Experiments

This script generates sample files for dynamic quantum walk experiments with angle noise.
It creates the sample data that will later be used to compute probability distributions.

Key Features:
- Multi-process execution (one process per deviation value)
- Dynamic angle noise implementation using random_angle_deviation
- Comprehensive logging for each process
- Graceful error handling and recovery
- Smart directory structure management

Usage:
    python generate_dynamic_samples.py

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
N = 20000              # System size (production scale for cluster)
steps = N//4           # Time steps (5000 for N=20000)
samples = 40           # Samples per deviation (full production count)
base_theta = math.pi/4 # Base theta parameter for dynamic angle noise

# Deviation values - Dynamic noise format (angle deviations) - Matching original static experiment
devs = [
    0,                  # No noise (equivalent to (0,0))
    0.2,                # Small noise (equivalent to (0, 0.2))
    0.6,                # Medium noise (equivalent to (0, 0.6))
    0.8,                # Medium noise (equivalent to (0, 0.8))
    1.0,                # Large noise (equivalent to (0, 1))
]

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples_dynamic"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "dynamic_sample_generation")

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())  # Use all available cores for cluster
PROCESS_TIMEOUT = 7200  # 2 hours timeout per process for large N

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "dynamic_sample_generation", "dynamic_sample_generation_master.log")

# Global shutdown flag
SHUTDOWN_REQUESTED = False

# Initial state configuration
initial_state_kwargs = {"nodes": [N//2]}

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}{theta_str}_dynamic_samples.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_dynamic_samples")
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
# DYNAMIC EXPERIMENT DIRECTORY FUNCTIONS
# ============================================================================

def get_dynamic_experiment_dir(tesselation_func, has_noise, N, noise_params=None, base_dir="experiments_data_samples_dynamic", base_theta=None):
    """
    Get experiment directory for dynamic noise experiments.
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

def dummy_tesselation_func(N):
    """Dummy tessellation function for dynamic noise (tessellations are built-in)"""
    return None

# ============================================================================
# DYNAMIC SAMPLE GENERATION FUNCTIONS
# ============================================================================

def setup_process_environment(dev, process_id, base_theta_param):
    """Setup imports, logging and basic environment for a worker process"""
    import numpy as np
    import networkx as nx
    
    dev_rounded = round(dev, 6)
    dev_str = f"{dev_rounded:.6f}"
    logger, log_file = setup_process_logging(dev_str, process_id, base_theta_param)
    
    return logger, log_file, np, nx

def log_process_startup(logger, dev, N, steps, samples_count, base_theta_param):
    """Log process startup information and parameters"""
    logger.info(f"=== DYNAMIC SAMPLE GENERATION STARTED ===")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Deviation: {dev}")
    logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, base_theta={base_theta_param:.6f}")
    
    if dev == 0:
        logger.info(f"[DEV=0 CASE] No angle noise: Perfect deterministic evolution")
        logger.info(f"[DEV=0 CASE] base_theta = {base_theta_param:.10f} radians = {base_theta_param/math.pi:.6f}*pi")
    else:
        logger.info(f"[ANGLE NOISE] Random angle deviations with dev={dev:.6f}")
        logger.info(f"[ANGLE NOISE] base_theta = {base_theta_param:.10f} radians = {base_theta_param/math.pi:.6f}*pi")

def setup_experiment_environment(dev, N, base_theta_param):
    """Setup experiment directory and import required modules"""
    from sqw.experiments_expanded import running
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    from sqw.utils import random_angle_deviation
    
    has_noise = dev > 0
    noise_params = [dev]
    exp_dir = get_dynamic_experiment_dir(
        dummy_tesselation_func, has_noise, N, 
        noise_params=noise_params, 
        base_dir=SAMPLES_BASE_DIR, 
        base_theta=base_theta_param
    )
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir, running, even_line_two_tesselation, uniform_initial_state, random_angle_deviation

def check_sample_exists(exp_dir, steps, sample_idx):
    """Check if all files for a sample already exist"""
    for step_idx in range(steps):
        step_dir = os.path.join(exp_dir, f"step_{step_idx}")
        filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
        filepath = os.path.join(step_dir, filename)
        if not os.path.exists(filepath):
            return False
    return True

def generate_sample_angles(dev, base_theta_param, steps, random_angle_deviation):
    """Generate angles for a single sample based on deviation parameters"""
    if dev == 0:
        return [[base_theta_param, base_theta_param]] * steps
    else:
        return random_angle_deviation([base_theta_param, base_theta_param], [dev, dev], steps)

def create_step_saver(exp_dir, sample_idx, steps, logger):
    """Create a callback function for saving individual steps"""
    def save_step_callback(step_idx, state):
        step_dir = os.path.join(exp_dir, f"step_{step_idx}")
        os.makedirs(step_dir, exist_ok=True)
        
        filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
        filepath = os.path.join(step_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        del state
        gc.collect()
        
        if step_idx % 100 == 0 or step_idx == steps - 1:
            logger.info(f"    Saving step {step_idx}/{steps - 1} (quantum walk state for time step {step_idx})")
    
    return save_step_callback

def run_single_sample_simulation(N, steps, angles, initial_nodes, running, even_line_two_tesselation, uniform_initial_state, nx):
    """Execute the quantum walk simulation for a single sample"""
    tesselation_order = [[0, 1] for x in range(steps)]
    
    G = nx.cycle_graph(N)
    T = even_line_two_tesselation(N)
    initial_state = uniform_initial_state(N, nodes=initial_nodes)
    
    evolution_states = running(G, T, steps, initial_state, angles, tesselation_order)
    return evolution_states

def process_single_sample(sample_idx, samples_count, exp_dir, steps, dev, base_theta_param, 
                         initial_nodes, logger, running, even_line_two_tesselation, 
                         uniform_initial_state, random_angle_deviation, N, nx):
    """Process computation and saving for a single sample"""
    sample_start_time = time.time()
    
    if check_sample_exists(exp_dir, steps, sample_idx):
        logger.info(f"Sample {sample_idx+1}/{samples_count} already exists, skipping...")
        return False, 0
    
    logger.info(f"Computing sample {sample_idx+1}/{samples_count}...")
    
    angles = generate_sample_angles(dev, base_theta_param, steps, random_angle_deviation)
    save_step_callback = create_step_saver(exp_dir, sample_idx, steps, logger)
    
    try:
        logger.debug(f"Starting dynamic quantum walk simulation for sample {sample_idx}")
        
        evolution_states = run_single_sample_simulation(
            N, steps, angles, initial_nodes, running, 
            even_line_two_tesselation, uniform_initial_state, nx
        )
        
        for step_idx, state in enumerate(evolution_states):
            save_step_callback(step_idx, state)
        
        logger.debug(f"Dynamic quantum walk simulation completed for sample {sample_idx} ({steps} time steps saved)")
        
    except MemoryError as mem_error:
        logger.error(f"Memory error in sample {sample_idx}: {mem_error}")
        logger.error(f"Try reducing N or running with fewer parallel processes")
        raise
    except Exception as comp_error:
        logger.error(f"Computation error in sample {sample_idx}: {comp_error}")
        logger.error(traceback.format_exc())
        raise
    
    sample_time = time.time() - sample_start_time
    gc.collect()
    
    logger.info(f"Sample {sample_idx+1}/{samples_count} completed in {sample_time:.1f}s")
    return True, sample_time

def create_success_result(dev, process_id, computed_samples, skipped_samples, samples_count, total_time, log_file):
    """Create a success result dictionary"""
    return {
        "dev": dev,
        "process_id": process_id,
        "computed_samples": computed_samples,
        "skipped_samples": skipped_samples,
        "total_samples": samples_count,
        "total_time": total_time,
        "log_file": log_file,
        "success": True,
        "error": None
    }

def create_error_result(dev, process_id, samples_count, log_file, error_msg):
    """Create an error result dictionary"""
    return {
        "dev": dev,
        "process_id": process_id,
        "computed_samples": 0,
        "skipped_samples": 0,
        "total_samples": samples_count,
        "total_time": 0,
        "log_file": log_file,
        "success": False,
        "error": error_msg
    }

def generate_dynamic_samples_for_dev(dev_args):
    """Main worker function that orchestrates dynamic sample generation for a single deviation"""
    dev, process_id, N, steps, samples_count, base_theta_param, initial_state_kwargs = dev_args
    
    logger, log_file, np, nx = setup_process_environment(dev, process_id, base_theta_param)
    
    try:
        log_process_startup(logger, dev, N, steps, samples_count, base_theta_param)
        
        exp_dir, running, even_line_two_tesselation, uniform_initial_state, random_angle_deviation = setup_experiment_environment(
            dev, N, base_theta_param
        )
        
        logger.info(f"Experiment directory: {exp_dir}")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        dev_skipped_samples = 0
        initial_nodes = initial_state_kwargs.get('nodes', [])
        
        for sample_idx in range(samples_count):
            sample_computed, sample_time = process_single_sample(
                sample_idx, samples_count, exp_dir, steps, dev, base_theta_param,
                initial_nodes, logger, running, even_line_two_tesselation,
                uniform_initial_state, random_angle_deviation, N, nx
            )
            
            if sample_computed:
                dev_computed_samples += 1
            else:
                dev_skipped_samples += 1
            
            # Progress summary every 5 samples or at the end
            if (sample_idx + 1) % 5 == 0 or sample_idx == samples_count - 1:
                logger.info(f"Sample progress: {sample_idx + 1}/{samples_count} processed ({dev_computed_samples} computed, {dev_skipped_samples} skipped)")
        
        dev_time = time.time() - dev_start_time
        dev_rounded = round(dev, 6)
        dev_str = f"{dev_rounded:.4f}"
        
        logger.info(f"=== DYNAMIC SAMPLE GENERATION COMPLETED ===")
        logger.info(f"Deviation {dev_str}: {dev_computed_samples} computed, {dev_skipped_samples} skipped")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        return create_success_result(dev, process_id, dev_computed_samples, dev_skipped_samples, samples_count, dev_time, log_file)
        
    except Exception as e:
        dev_rounded = round(dev, 6)
        dev_str = f"{dev_rounded:.4f}"
        error_msg = f"Error in dynamic sample generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return create_error_result(dev, process_id, samples_count, log_file, error_msg)

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for dynamic sample generation."""
    
    print("=== DYNAMIC QUANTUM WALK SAMPLE GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"Output directory: {SAMPLES_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== DYNAMIC QUANTUM WALK SAMPLE GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, base_theta={base_theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, base_theta, initial_state_kwargs)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for dynamic sample generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_dynamic_samples_for_dev, args)
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
                                         f"Computed: {result['computed_samples']}, "
                                         f"Skipped: {result['skipped_samples']}, "
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
                        "computed_samples": 0, "skipped_samples": 0, "total_time": 0
                    })
                except Exception as e:
                    dev_rounded = round(dev, 6)
                    error_msg = f"Dev {dev_rounded:.4f}: EXCEPTION - {str(e)}"
                    master_logger.error(error_msg)
                    process_results.append({
                        "dev": dev, "success": False, "error": error_msg,
                        "computed_samples": 0, "skipped_samples": 0, "total_time": 0
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
    total_computed = sum(r.get("computed_samples", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_samples", 0) for r in process_results)
    
    print(f"\n=== DYNAMIC GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Samples: {total_computed} computed, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== DYNAMIC GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Samples: {total_computed} computed, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Dev {result['dev']}: SUCCESS - "
                             f"Computed: {result.get('computed_samples', 0)}, "
                             f"Skipped: {result.get('skipped_samples', 0)}, "
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
