#!/usr/bin/env python3

"""
Generate Sample Files for Quantum Walk Experiments - Linspace Version

This script generates sample files for quantum walk experiments with configurable parameters.
It creates the sample data that will later be used to compute probability distributions.

Key Features:
- Configurable number of processes (each handling multiple deviation values)
- Memory-efficient sparse matrix computation
- Comprehensive logging for each process
- Graceful error handling and recovery
- Smart directory structure management

Usage:
    python generate_samples_linspace.py

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
NUM_PROCESSES = 10        # Number of processes to use (CONFIGURABLE)
PROCESS_TIMEOUT = 3600   # 1 hour timeout per process

# Directory configuration
SAMPLES_BASE_DIR = "experiments_data_samples_linspace"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_samples_linspace")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_samples_linspace", "sample_generation_master.log")

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process{process_id}_{dev_range_str}{theta_str}_samples.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"process_{process_id}")
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
# SAMPLE GENERATION FUNCTIONS
# ============================================================================

def generate_samples_for_dev_chunk(chunk_args):
    """
    Worker function to generate samples for a chunk of deviation values in a separate process.
    
    Args:
        chunk_args: Tuple containing (dev_chunk, process_id, N, steps, samples, theta, initial_state_kwargs)
    
    Returns:
        dict: Results from the sample generation process
    """
    dev_chunk, process_id, N, steps, samples_count, theta_param, initial_state_kwargs = chunk_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(process_id, dev_chunk, theta_param)
    
    try:
        logger.info(f"=== SAMPLE GENERATION STARTED ===")
        logger.info(f"PID: {os.getpid()}")
        logger.info(f"Process ID: {process_id}")
        logger.info(f"Deviation chunk size: {len(dev_chunk)}")
        if len(dev_chunk) > 0:
            first_dev = dev_chunk[0][1] if isinstance(dev_chunk[0], tuple) else dev_chunk[0]
            last_dev = dev_chunk[-1][1] if isinstance(dev_chunk[-1], tuple) else dev_chunk[-1]
            logger.info(f"Deviation range: {first_dev:.4f} to {last_dev:.4f}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples_count}, theta={theta_param:.6f}")
        
        # Import the memory-efficient sparse implementation
        from sqw.experiments_sparse import running_streaming_sparse
        
        chunk_start_time = time.time()
        chunk_computed_samples = 0
        chunk_skipped_samples = 0
        processed_devs = 0
        
        # Process each deviation in the chunk
        for dev in dev_chunk:
            dev_start_time = time.time()
            logger.info(f"Processing deviation {processed_devs+1}/{len(dev_chunk)}: {dev}")
            
            # VALIDATION FOR DEV=0 CASE
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_min, dev_max = dev
                if dev_min == 0 and dev_max == 0:
                    logger.info(f"[DEV=0 CASE] No noise: Perfect deterministic evolution")
                    logger.info(f"[DEV=0 CASE] theta = {theta_param:.10f} radians = {theta_param/math.pi:.6f}*pi")
            elif dev == 0:
                logger.info(f"[DEV=0 CASE] Legacy format: Perfect deterministic evolution - no noise")
                logger.info(f"[DEV=0 CASE] theta = {theta_param:.10f} radians = {theta_param/math.pi:.6f}*pi")
            
            # Setup experiment directory - handle new deviation format
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                # Direct (minVal, maxVal) format
                min_val, max_val = dev
                has_noise = max_val > 0
            else:
                # Single value format
                has_noise = dev > 0
            
            # Create custom directory structure for linspace experiment
            from smart_loading_static import format_theta_for_directory
            
            # Format deviation for directory name
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_min, dev_max = dev
                dev_folder = f"dev_min{dev_min:.3f}_max{dev_max:.3f}"
            else:
                dev_folder = f"dev_{dev:.3f}"
            
            # Create directory structure: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            theta_folder = format_theta_for_directory(theta_param)
            exp_dir = os.path.join(SAMPLES_BASE_DIR, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            os.makedirs(exp_dir, exist_ok=True)
            
            logger.info(f"Experiment directory: {exp_dir}")
            
            # VALIDATION: Check that directory path includes theta correctly
            expected_theta_folder = format_theta_for_directory(theta_param)
            if expected_theta_folder and expected_theta_folder not in exp_dir:
                logger.warning(f"[DIRECTORY WARNING] Expected theta folder '{expected_theta_folder}' not found in path!")
                logger.warning(f"[DIRECTORY WARNING] This might cause data mixing between different theta values!")
            else:
                logger.info(f"[DIRECTORY OK] Theta folder '{expected_theta_folder}' correctly included in path")
                logger.info(f"[DIRECTORY OK] Using custom static_noise_linspace structure")
            
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
                    logger.info(f"  Sample {sample_idx+1}/{samples_count} already exists, skipping...")
                    dev_skipped_samples += 1
                    continue
                
                logger.info(f"  Computing sample {sample_idx+1}/{samples_count}...")
                
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
                        logger.info(f"      Saved step {step_idx}/{steps}")
                
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
                
                logger.info(f"  Sample {sample_idx+1}/{samples_count} completed in {sample_time:.1f}s")
            
            dev_time = time.time() - dev_start_time
            processed_devs += 1
            chunk_computed_samples += dev_computed_samples
            chunk_skipped_samples += dev_skipped_samples
            
            # Format dev for display
            if isinstance(dev, tuple):
                dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            
            logger.info(f"Deviation {dev_str} completed: {dev_computed_samples} computed, {dev_skipped_samples} skipped in {dev_time:.1f}s")
        
        chunk_time = time.time() - chunk_start_time
        
        logger.info(f"=== CHUNK GENERATION COMPLETED ===")
        logger.info(f"Process {process_id}: {processed_devs} deviations processed")
        logger.info(f"Total samples: {chunk_computed_samples} computed, {chunk_skipped_samples} skipped")
        logger.info(f"Total time: {chunk_time:.1f}s")
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": processed_devs,
            "computed_samples": chunk_computed_samples,
            "skipped_samples": chunk_skipped_samples,
            "total_samples": samples_count * len(dev_chunk),
            "total_time": chunk_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error in sample generation for process {process_id}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "process_id": process_id,
            "dev_chunk": dev_chunk,
            "processed_devs": 0,
            "computed_samples": 0,
            "skipped_samples": 0,
            "total_samples": samples_count * len(dev_chunk),
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
    
    print("=== QUANTUM WALK SAMPLE GENERATION - LINSPACE VERSION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    print(f"Total deviations: {len(devs)}")
    print(f"Multiprocessing: {NUM_PROCESSES} processes")
    print(f"Output directory: {SAMPLES_BASE_DIR}")
    print("=" * 60)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== QUANTUM WALK SAMPLE GENERATION - LINSPACE VERSION STARTED ===")
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
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev_chunk in enumerate(dev_chunks):
        if len(dev_chunk) > 0:  # Only create processes for non-empty chunks
            args = (dev_chunk, process_id, N, steps, samples, theta, initial_state_kwargs)
            process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for sample generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            # Submit all processes
            future_to_process = {}
            for args in process_args:
                future = executor.submit(generate_samples_for_dev_chunk, args)
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
                                         f"{result['computed_samples']} computed, "
                                         f"{result['skipped_samples']} skipped, "
                                         f"time: {result['total_time']:.1f}s")
                    else:
                        master_logger.error(f"Process {process_id} failed: {result['error']}")
                        
                except TimeoutError:
                    master_logger.error(f"Process {process_id} timed out after {PROCESS_TIMEOUT}s")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": "Timeout",
                        "processed_devs": 0, "computed_samples": 0, "skipped_samples": 0,
                        "total_samples": 0, "total_time": PROCESS_TIMEOUT
                    })
                except Exception as e:
                    master_logger.error(f"Process {process_id} crashed: {str(e)}")
                    process_results.append({
                        "process_id": process_id, "success": False, "error": str(e),
                        "processed_devs": 0, "computed_samples": 0, "skipped_samples": 0,
                        "total_samples": 0, "total_time": 0
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
    total_computed = sum(r.get("computed_samples", 0) for r in process_results)
    total_skipped = sum(r.get("skipped_samples", 0) for r in process_results)
    
    print(f"\n=== GENERATION SUMMARY ===")
    print(f"Total time: {total_time:.1f}s")
    print(f"Processes: {successful_processes} successful, {failed_processes} failed")
    print(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    print(f"Samples: {total_computed} computed, {total_skipped} skipped")
    print(f"Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
    
    master_logger.info("=== GENERATION SUMMARY ===")
    master_logger.info(f"Total time: {total_time:.1f}s")
    master_logger.info(f"Processes: {successful_processes} successful, {failed_processes} failed")
    master_logger.info(f"Deviations: {total_processed_devs}/{len(devs)} processed")
    master_logger.info(f"Samples: {total_computed} computed, {total_skipped} skipped")
    
    # Log individual process results
    master_logger.info("INDIVIDUAL PROCESS RESULTS:")
    for result in process_results:
        if result["success"]:
            master_logger.info(f"  Process {result['process_id']}: SUCCESS - "
                             f"Devs: {result.get('processed_devs', 0)}, "
                             f"Computed: {result.get('computed_samples', 0)}, "
                             f"Skipped: {result.get('skipped_samples', 0)}, "
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
