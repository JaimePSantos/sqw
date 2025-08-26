#!/usr/bin/env python3

"""
Generate Sample Files for Quantum Walk Experiments

This script generates sample files for quantum walk experiments with configurable parameters.
It creates the sample data that will later be used to compute probability distributions.

Key Features:
- Multi-process execution (one process per deviation value)
- Memory-efficient sparse matrix computation
- Comprehensive logging for each process
- Graceful error handling and recovery
- Smart directory structure management

Usage:
    python generate_samples.py

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
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from datetime import datetime
import traceback
import signal
import math
import numpy as np
import gc

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 20000                # System size (small for testing)
steps = N//4           # Time steps (25 for N=100)
samples = 40         # Samples per deviation (small for testing)
theta = math.pi/3      # Theta parameter for static noise
# samples = 5          # Samples per deviation (small for testing)
# theta = math.pi/4     # Theta parameter for static noise

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

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "sample_generation")

# Multiprocessing configuration
MAX_PROCESSES = min(len(devs), mp.cpu_count())
PROCESS_TIMEOUT = 3600  # 1 hour timeout per process

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "sample_generation", "sample_generation_master.log")

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

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_process_logging(dev_value, process_id, theta=None):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format dev_value for filename
    if isinstance(dev_value, (tuple, list)) and len(dev_value) == 2:
        min_val, max_val = dev_value
        dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        dev_str = f"{float(dev_value):.3f}"
    
    # Format theta for filename
    if theta is not None:
        theta_str = f"_theta{theta:.6f}"
    else:
        theta_str = ""
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}{theta_str}_samples.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}_samples")
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
# SAMPLE GENERATION FUNCTIONS
# ============================================================================

def generate_samples_for_dev(dev_args):
    """
    Worker function to generate samples for a single deviation value in a separate process.
    
    Args:
        dev_args: Tuple containing (dev, process_id, N, steps, samples, theta, initial_state_kwargs)
    
    Returns:
        dict: Results from the sample generation process
    """
    dev, process_id, N, steps, samples_count, theta_param, initial_state_kwargs = dev_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id, theta_param)
    
    try:
        logger.info(f"=== SAMPLE GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Deviation: {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        
        # VALIDATION FOR DEV=0 CASE
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            dev_min, dev_max = dev
            if dev_min == 0 and dev_max == 0:
                logger.info(f"[DEV=0 CASE] No noise: Perfect deterministic evolution")
                logger.info(f"[DEV=0 CASE] theta = {theta_param:.10f} radians = {theta_param/math.pi:.6f}*pi")
        elif dev == 0:
            logger.info(f"[DEV=0 CASE] Legacy format: Perfect deterministic evolution - no noise")
            logger.info(f"[DEV=0 CASE] theta = {theta_param:.10f} radians = {theta_param/math.pi:.6f}*pi")
        
        # Import the memory-efficient sparse implementation
        from sqw.experiments_sparse import running_streaming_sparse
        from smart_loading_static import get_experiment_dir
        
        # Setup experiment directory - handle new deviation format
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # Direct (minVal, maxVal) format
            min_val, max_val = dev
            has_noise = max_val > 0
        else:
            # Single value format
            has_noise = dev > 0
        
        # With unified structure, we always include noise_params (including 0 for no noise)
        noise_params = [dev]
        exp_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type="static_noise", 
                                   base_dir=SAMPLES_BASE_DIR, theta=theta_param)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"Experiment directory: {exp_dir}")
        
        # VALIDATION: Check that directory path includes theta correctly
        from smart_loading_static import format_theta_for_directory
        expected_theta_folder = format_theta_for_directory(theta_param)
        if expected_theta_folder and expected_theta_folder not in exp_dir:
            logger.warning(f"[DIRECTORY WARNING] Expected theta folder '{expected_theta_folder}' not found in path!")
            logger.warning(f"[DIRECTORY WARNING] This might cause data mixing between different theta values!")
        else:
            logger.info(f"[DIRECTORY OK] Theta folder '{expected_theta_folder}' correctly included in path")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        dev_skipped_samples = 0
        
        for sample_idx in range(samples_count):
            sample_start_time = time.time()
            
            # Check if this sample already exists (all step files) - use same logic as static cluster
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                logger.info(f"Sample {sample_idx+1}/{samples_count} already exists, skipping...")
                dev_skipped_samples += 1
                continue
            
            logger.info(f"Computing sample {sample_idx+1}/{samples_count}...")
            
            # Extract initial nodes from initial_state_kwargs
            initial_nodes = initial_state_kwargs.get('nodes', [])
            
            # Create step callback function for streaming saves
            def save_step_callback(step_idx, state):
                """Callback function to save each step as it's computed"""
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                os.makedirs(step_dir, exist_ok=True)
                
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(state, f)
                
                # Explicitly delete state reference to help memory management
                del state
                gc.collect()
                
                # Progress logging for large computations
                if step_idx % 100 == 0 or step_idx == steps:
                    logger.info(f"    Saved step {step_idx}/{steps}")
            
            # Memory-efficient streaming approach
            try:
                logger.debug(f"Starting quantum walk simulation for sample {sample_idx}")
                
                # Call the sparse streaming function
                final_state = running_streaming_sparse(
                    N, theta_param, steps,
                    initial_nodes=initial_nodes,
                    deviation_range=dev,
                    step_callback=save_step_callback
                )
                
                logger.debug(f"Quantum walk simulation completed for sample {sample_idx}")
                
            except MemoryError as mem_error:
                logger.error(f"Memory error in sample {sample_idx}: {mem_error}")
                logger.error(f"Try reducing N or running with fewer parallel processes")
                raise
            except Exception as comp_error:
                logger.error(f"Computation error in sample {sample_idx}: {comp_error}")
                logger.error(traceback.format_exc())
                raise
            
            dev_computed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info(f"Sample {sample_idx+1}/{samples_count} completed in {sample_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        
        logger.info(f"=== SAMPLE GENERATION COMPLETED ===")
        logger.info(f"Deviation {dev_str}: {dev_computed_samples} computed, {dev_skipped_samples} skipped")
        logger.info(f"Total time: {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": dev_computed_samples,
            "skipped_samples": dev_skipped_samples,
            "total_samples": samples_count,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        error_msg = f"Error in sample generation for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
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

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """Main execution function for sample generation."""
    
    print("=== QUANTUM WALK SAMPLE GENERATION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation values: {devs}")
    print(f"Multiprocessing: {MAX_PROCESSES} processes")
    print(f"Output directory: {SAMPLES_BASE_DIR}")
    print("=" * 50)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== QUANTUM WALK SAMPLE GENERATION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    
    start_time = time.time()
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, theta, initial_state_kwargs)
        process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for sample generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
            # Submit all processes
            future_to_dev = {}
            for args in process_args:
                future = executor.submit(generate_samples_for_dev, args)
                future_to_dev[future] = args[0]  # dev value
            
            # Collect results with timeout handling
            for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT * len(devs)):
                dev = future_to_dev[future]
                
                try:
                    result = future.result(timeout=PROCESS_TIMEOUT)
                    process_results.append(result)
                    
                    if result["success"]:
                        master_logger.info(f"Process for dev {dev} completed successfully: "
                                         f"{result['computed_samples']} computed, "
                                         f"{result['skipped_samples']} skipped, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process for dev {dev} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process for dev {dev} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "dev": dev, "success": False, "error": "Timeout",
                        "computed_samples": 0, "skipped_samples": 0,
                        "total_samples": samples, "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process for dev {dev} crashed: {str(e)}")
                    process_results.append({
                        "dev": dev, "success": False, "error": str(e),
                        "computed_samples": 0, "skipped_samples": 0,
                        "total_samples": samples, "total_time": 0
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
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Samples: {total_computed} computed, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
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
