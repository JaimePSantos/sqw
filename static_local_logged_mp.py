#!/usr/bin/env python3

"""
Static noise experiment for quantum walks.

This script runs static noise experiments for quantum walks with configurable parameters.

Now uses smart loading from smart_loading_static module.

IMPROVED ROBUSTNESS AND LOGGING:
- Added proper timeout handling for mean probability computation phase
- Enhanced logging with progress updates, resource monitoring, and ETA calculation
- Added graceful shutdown handling with signal handlers (SIGINT, SIGTERM, SIGHUP)
- Improved error recovery and partial result handling
- Added system resource monitoring (memory, CPU usage)
- More frequent progress updates with timestamps
- Better error messages and troubleshooting information

Execution Modes:
1. Full Pipeline (default): Compute samples + analysis + plots + archive
2. Samples Only: Set CALCULATE_SAMPLES_ONLY = True to only compute and save samples
3. Analysis Only: Set SKIP_SAMPLE_COMPUTATION = True to skip sample computation
4. Custom: Adjust individual toggles for plotting, archiving, etc.

This modular approach allows you to:
- Run expensive sample computation on cluster, then analysis locally
- Recompute analysis with different parameters without recomputing samples
- Split long computations into manageable chunks
"""

# ============================================================================
# MODULE LEVEL FUNCTIONS (for multiprocessing serialization)
# ============================================================================

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

import time
import math
import numpy as np
import os
import sys
import subprocess
import signal
import tarfile
import traceback
import pickle
import gc
import psutil
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import logging

# Import crash-safe logging decorator
from logging_module.crash_safe_logging import crash_safe_log

# ============================================================================
# SIGNAL HANDLING FOR GRACEFUL SHUTDOWN
# ============================================================================

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SHUTDOWN] Received signal {signum}. Initiating graceful shutdown...")
    print("[SHUTDOWN] Waiting for current processes to complete...")
    print("[SHUTDOWN] This may take a few minutes. Do not force-kill unless necessary.")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal (Unix)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Plotting switch
ENABLE_PLOTTING = True  # Set to False to disable plotting
USE_LOGLOG_PLOT = True  # Set to True to use log-log scale for plotting
PLOT_FINAL_PROBDIST = True  # Set to True to plot probability distributions at final time step
SAVE_FIGURES = True  # Set to False to disable saving figures to files

# Archive switch
CREATE_TAR_ARCHIVE = True  # Set to True to create tar archive of experiments_data_samples folder
USE_MULTIPROCESS_ARCHIVING = True  # Set to True to use multiprocess archiving for faster compression
MAX_ARCHIVE_PROCESSES = None  # Max processes for archiving (None = auto-detect)
EXCLUDE_SAMPLES_FROM_ARCHIVE = True  # Set to True to exclude raw sample files from archive (keeps only probDist and std)

# Computation control switches
CALCULATE_SAMPLES_ONLY = False  # Set to True to only compute and save samples (skip analysis)
SKIP_SAMPLE_COMPUTATION = False  # Set to True to skip sample computation (analysis only)

# Check for environment variable overrides (from safe_background_launcher.py)
if os.environ.get('ENABLE_PLOTTING'):
    ENABLE_PLOTTING = os.environ.get('ENABLE_PLOTTING').lower() == 'true'
if os.environ.get('CREATE_TAR_ARCHIVE'):
    CREATE_TAR_ARCHIVE = os.environ.get('CREATE_TAR_ARCHIVE').lower() == 'true'
if os.environ.get('USE_MULTIPROCESS_ARCHIVING'):
    USE_MULTIPROCESS_ARCHIVING = os.environ.get('USE_MULTIPROCESS_ARCHIVING').lower() == 'true'
if os.environ.get('MAX_ARCHIVE_PROCESSES'):
    try:
        MAX_ARCHIVE_PROCESSES = int(os.environ.get('MAX_ARCHIVE_PROCESSES'))
    except ValueError:
        pass
if os.environ.get('EXCLUDE_SAMPLES_FROM_ARCHIVE'):
    EXCLUDE_SAMPLES_FROM_ARCHIVE = os.environ.get('EXCLUDE_SAMPLES_FROM_ARCHIVE').lower() == 'true'
if os.environ.get('USE_MULTIPROCESS_MEAN_PROB'):
    USE_MULTIPROCESS_MEAN_PROB = os.environ.get('USE_MULTIPROCESS_MEAN_PROB').lower() == 'true'
if os.environ.get('MAX_MEAN_PROB_PROCESSES'):
    try:
        MAX_MEAN_PROB_PROCESSES = int(os.environ.get('MAX_MEAN_PROB_PROCESSES'))
    except ValueError:
        pass
if os.environ.get('CALCULATE_SAMPLES_ONLY'):
    CALCULATE_SAMPLES_ONLY = os.environ.get('CALCULATE_SAMPLES_ONLY').lower() == 'true'
if os.environ.get('SKIP_SAMPLE_COMPUTATION'):
    SKIP_SAMPLE_COMPUTATION = os.environ.get('SKIP_SAMPLE_COMPUTATION').lower() == 'true'

# Background execution switch - SAFER IMPLEMENTATION
RUN_IN_BACKGROUND = False  # Set to True to automatically run the process in background

# Check if background execution has been disabled externally
if os.environ.get('RUN_IN_BACKGROUND') == 'False':
    RUN_IN_BACKGROUND = False
BACKGROUND_LOG_FILE = "static_experiment_multiprocessing.log"  # Log file for background execution
BACKGROUND_PID_FILE = "static_experiment_mp.pid"  # PID file to track background process

# Experiment parameters
N = 200  # System size 
steps = N//6  # Time steps
samples = 50  # Samples per deviation

# Resource monitoring and management
print(f"[COMPUTATION SCALE] N={N}, steps={steps}, samples={samples}")
print(f"[STREAMING MODE] Memory-efficient sparse matrix computation saves states incrementally")

# Estimate computational requirements
total_qw_simulations = len([(0,0), (0, 0.2), (0, 0.6), (0, 0.8), (0, 1)]) * samples
estimated_time_per_sim = (N * steps) / 1000000  # rough estimate in minutes
total_estimated_time = total_qw_simulations * estimated_time_per_sim

print(f"[RESOURCE ESTIMATE] Total quantum walks: {total_qw_simulations}")
print(f"[RESOURCE ESTIMATE] Estimated time: {total_estimated_time:.1f} minutes")

if N > 10000 and steps > 1000:
    print(f"[WARNING] LARGE COMPUTATION: This will take significant time")
    print("   Consider running with fewer samples first to test")
    print("   Each process uses sparse matrix streaming to minimize memory usage")
    print(f"   Estimated memory per process: ~{2.0:.1f}MB (sparse matrices + state)")  # From our test

# Monitor initial system resources
try:
    import psutil
    memory = psutil.virtual_memory()
    print(f"[SYSTEM] Available memory: {memory.available / (1024**3):.1f}GB")
    print(f"[SYSTEM] CPU count: {mp.cpu_count()}")
except ImportError:
    print("[SYSTEM] psutil not available - install for resource monitoring")

# Environment variable overrides removed - samples value is set at top of file

if os.environ.get('FORCE_N_VALUE'):
    try:
        forced_N = int(os.environ.get('FORCE_N_VALUE'))
        print(f"[FORCED] Using N = {forced_N} (launcher override)")
        N = forced_N
        steps = N//4  # Recalculate steps
    except ValueError:
        pass

# Quantum walk parameters (for static noise, we only need theta)
theta = math.pi/3  # Base theta parameter for static noise
initial_state_kwargs = {"nodes": [N//2]}

# Deviation values for static noise experiments
# CUSTOMIZE THIS LIST: Add or remove deviation values as needed
# Format options:
# 1. Single value: devs = [0, 0.1, 0.5] - backward compatibility (range [0, value])
# 2. Tuple (minVal, maxVal): devs = [(0.0, 0.2)] - direct range format
# 3. Mixed: devs = [0, (0.0, 0.2), 0.5] - can mix formats
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),           # Medium noise range  
]   

# Multiprocessing configuration - Conservative for cluster stability
# For large computations, use fewer processes to avoid resource exhaustion
cpu_count = mp.cpu_count()
if N > 10000:
    # Use fewer processes for very large problems to avoid memory/time limits
    MAX_PROCESSES = min(len(devs), max(1, cpu_count // 2))
    print(f"[CONSERVATIVE] Using {MAX_PROCESSES} processes (half of {cpu_count} CPUs) for large N={N}")
else:
    MAX_PROCESSES = min(len(devs), cpu_count)
    print(f"[STANDARD] Using {MAX_PROCESSES} processes out of {cpu_count} CPUs")

PROCESS_LOG_DIR = "process_logs"  # Directory for individual process logs

# Timeout configuration - Scale with problem size
# Base timeout for each process, scaled by N and steps
BASE_TIMEOUT_PER_SAMPLE = 30  # seconds per sample for small N
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(3600, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 1 hour

# Mean probability timeout is typically longer since it processes all steps at once
MEAN_PROB_TIMEOUT_MULTIPLIER = 2.0  # Mean prob takes longer than sample computation
MEAN_PROB_TIMEOUT = max(7200, int(PROCESS_TIMEOUT * MEAN_PROB_TIMEOUT_MULTIPLIER))  # Minimum 2 hours

print(f"[TIMEOUT] Sample process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Mean prob process timeout: {MEAN_PROB_TIMEOUT} seconds ({MEAN_PROB_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")

# Mean probability multiprocessing configuration
USE_MULTIPROCESS_MEAN_PROB = True  # Set to True to use multiprocessing for mean probability calculation
MAX_MEAN_PROB_PROCESSES = min(5, MAX_PROCESSES)  # Don't exceed main process limit

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
        
        # Check for concerning resource usage
        if memory.percent > 90:
            warning_msg = f"{prefix} WARNING: High memory usage ({memory.percent:.1f}%)"
            if logger:
                logger.warning(warning_msg)
        
        if cpu_percent > 95:
            warning_msg = f"{prefix} WARNING: High CPU usage ({cpu_percent:.1f}%)"
            if logger:
                logger.warning(warning_msg)
            
    except ImportError:
        msg = f"{prefix} psutil not available - cannot monitor resources"
        if logger:
            logger.info(msg)
    except Exception as e:
        msg = f"{prefix} Error monitoring resources: {e}"
        if logger:
            logger.error(msg)

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

# ============================================================================
# MULTIPROCESSING LOGGING SETUP
# ============================================================================

def setup_process_logging(dev_value, process_id):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format dev_value for filename (handle both old and new formats)
    if isinstance(dev_value, str):
        dev_str = dev_value  # Already formatted as string
    elif isinstance(dev_value, (tuple, list)) and len(dev_value) == 2:
        # Direct (minVal, maxVal) format
        min_val, max_val = dev_value
        dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        # Single value format
        dev_str = f"{float(dev_value):.3f}"
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_pid_{process_id}.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"dev_{dev_str}")
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
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [PID:%(process)d] [DEV:%(name)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def setup_master_logging():
    """Setup logging for the master process"""
    master_log_filename = "static_experiment_multiprocess.log"
    
    # Create master logger
    master_logger = logging.getLogger("master")
    master_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in master_logger.handlers[:]:
        master_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(master_log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [MASTER] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    master_logger.addHandler(file_handler)
    master_logger.addHandler(console_handler)
    
    return master_logger, master_log_filename

# ============================================================================
# RESOURCE MONITORING AND RECOVERY FUNCTIONS
# ============================================================================

def monitor_system_resources():
    """Monitor system memory and CPU usage"""
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent
        }
    except:
        return {'memory_percent': 0, 'memory_available_gb': 0, 'cpu_percent': 0}

def check_sample_exists(exp_dir, sample_id):
    """Check if a specific sample already exists"""
    sample_file = os.path.join(exp_dir, f"sample_{sample_id}.pkl")
    return os.path.exists(sample_file)

def get_completed_samples(exp_dir, total_samples):
    """Get list of completed samples to enable resuming"""
    completed = []
    for i in range(total_samples):
        if check_sample_exists(exp_dir, i):
            completed.append(i)
    return completed

def log_resource_usage(logger, prefix=""):
    """Log current resource usage"""
    try:
        resources = monitor_system_resources()
        logger.info(f"{prefix}Resource usage: Memory {resources['memory_percent']:.1f}%, "
                   f"Available {resources['memory_available_gb']:.1f}GB, "
                   f"CPU {resources['cpu_percent']:.1f}%")
    except Exception as e:
        logger.warning(f"Could not monitor resources: {e}")

# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================

# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================

def compute_mean_probability_for_dev(dev_args):
    """Worker function to compute mean probability distributions for a single deviation value in a separate process"""
    dev, process_id, N, steps, samples, source_base_dir, target_base_dir, noise_type, theta, tesselation_func = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(f"meanprob_{dev_str}", process_id)
    
    try:
        logger.info(f"Starting mean probability computation for deviation {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}")
        
        # Import required modules
        import pickle
        import gc
        from sqw.states import amp2prob
        from smart_loading_static import find_experiment_dir_flexible, get_experiment_dir
        
        dev_start_time = time.time()
        
        # Handle deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # Direct (minVal, maxVal) format
            min_val, max_val = dev
            has_noise = max_val > 0
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            # Single value format
            has_noise = dev > 0
            dev_str = f"{dev:.3f}"
        
        if noise_type == "static_noise":
            noise_params = [dev]
            param_name = "static_dev"
        
        # Get source and target directories
        if noise_type == "static_noise":
            source_exp_dir, found_format = find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=source_base_dir, theta=theta)
            # For target directory (probDist), always use new structure with samples
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=target_base_dir, theta=theta, samples=samples)
        else:
            source_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=source_base_dir, theta=theta)
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=target_base_dir, theta=theta, samples=samples)
        
        os.makedirs(target_exp_dir, exist_ok=True)
        logger.info(f"Processing {steps} steps for {param_name}={dev_str}")
        logger.info(f"Source: {source_exp_dir}")
        logger.info(f"Target: {target_exp_dir}")
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        
        processed_steps = 0
        last_log_time = time.time()
        
        for step_idx in range(steps):
            # Check if mean probability file already exists
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(target_exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                processed_steps += 1
                if step_idx % 100 == 0:
                    logger.info(f"    Step {step_idx+1}/{steps} already exists, skipping")
                continue
            
            # Log progress more frequently and monitor resources
            current_time = time.time()
            # Log every 100 steps, but only log time-based updates if it's been more than 5 minutes
            # AND we're not already logging for the 100-step interval
            should_log_progress = (step_idx % 100 == 0)
            should_log_resources = (current_time - last_log_time >= 300)  # Every 5 minutes
            
            if should_log_progress:
                logger.info(f"    Step {step_idx+1}/{steps} processing... (processed: {processed_steps})")
            
            if should_log_resources:
                log_system_resources(logger, "[WORKER]")
                last_log_time = current_time
            
            step_dir = os.path.join(source_exp_dir, f"step_{step_idx}")
            
            # Optimized streaming processing - load and process samples one at a time
            mean_prob_dist = None
            valid_samples = 0
            
            for sample_idx in range(samples):
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                if os.path.exists(filepath):
                    # Load sample
                    with open(filepath, "rb") as f:
                        state = pickle.load(f)
                    
                    # Convert to probability distribution
                    prob_dist = amp2prob(state)  # |amplitude|^2
                    
                    # Update running mean
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
                with open(mean_filepath, "wb") as f:
                    pickle.dump(mean_prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                processed_steps += 1
                
                # Only log completion every 100 steps or on final step
                if step_idx % 100 == 0 or step_idx == steps - 1:
                    logger.info(f"    Step {step_idx+1}/{steps} processed (valid samples: {valid_samples})")
            else:
                # Only log missing samples every 100 steps to avoid spam
                if step_idx % 100 == 0:
                    logger.warning(f"No valid samples found for step {step_idx+1}")
            
            # Force garbage collection periodically to keep memory usage low
            if step_idx % 50 == 0:
                gc.collect()
        
        dev_time = time.time() - dev_start_time
        logger.info(f"Deviation {dev_str} completed: {processed_steps}/{steps} steps in {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "processed_steps": processed_steps,
            "total_steps": steps,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        error_msg = f"Error in mean probability process for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "processed_steps": 0,
            "total_steps": steps,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

def compute_dev_samples(dev_args):
    """Worker function to compute samples for a single deviation value in a separate process"""
    dev, process_id, N, steps, samples, theta, initial_state_kwargs = dev_args
    
    # Setup logging for this process - need to format dev for logging
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"Starting computation for deviation {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.4f}")
        
        # Import required modules (each process needs its own imports)
        # Import the memory-efficient sparse implementation
        from sqw.experiments_sparse import running_streaming_sparse
        from smart_loading_static import get_experiment_dir
        import pickle
        import gc  # For garbage collection
        
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
                                   base_dir="experiments_data_samples", theta=theta)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"Experiment directory: {exp_dir}")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        
        for sample_idx in range(samples):
            sample_start_time = time.time()
            
            # Check if this sample already exists (all step files)
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                if not os.path.exists(filepath):
                    sample_exists = False
                    break
            
            if sample_exists:
                logger.info(f"Sample {sample_idx+1}/{samples} already exists, skipping")
                dev_computed_samples += 1
                continue
            
            logger.info(f"Computing sample {sample_idx+1}/{samples}...")
            
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
                    # Log memory usage at checkpoints
                    resources = monitor_system_resources()
                    logger.info(f"    Memory usage: {resources['memory_percent']:.1f}%, Available: {resources['memory_available_gb']:.1f}GB")
            
            # Memory-efficient streaming approach
            try:
                # Log memory before starting computation
                resources_before = monitor_system_resources()
                logger.info(f"  Starting computation - Memory: {resources_before['memory_percent']:.1f}%, Available: {resources_before['memory_available_gb']:.1f}GB")
                
                # Run the quantum walk experiment using sparse streaming approach
                logger.info(f"  Running sparse streaming quantum walk: N={N}, steps={steps}, dev={dev}")
                
                final_state = running_streaming_sparse(
                    N, theta, steps,
                    initial_nodes=initial_nodes,
                    deviation_range=dev,
                    step_callback=save_step_callback
                )
                
                # Log memory after completion
                resources_after = monitor_system_resources()
                logger.info(f"  Sparse streaming computation completed, all {steps+1} steps saved incrementally")
                logger.info(f"  Final memory usage: {resources_after['memory_percent']:.1f}%, Available: {resources_after['memory_available_gb']:.1f}GB")
                
                # Explicitly delete final state reference and force cleanup
                del final_state
                gc.collect()
                
                # No need to explicitly delete anything else - streaming approach doesn't accumulate states
                
            except MemoryError as mem_error:
                logger.error(f"Memory error during computation: {mem_error}")
                logger.error("Consider reducing N, steps, or running fewer processes simultaneously")
                raise
            except Exception as comp_error:
                logger.error(f"Computation error: {comp_error}")
                raise
            
            dev_computed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info(f"Sample {sample_idx+1}/{samples} completed in {sample_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        logger.info(f"Deviation {dev_str} completed: {dev_computed_samples} samples in {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": dev_computed_samples,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        import traceback
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        error_msg = f"Error in process for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": 0,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }

# ============================================================================
# STANDARD DEVIATION DATA MANAGEMENT
# ============================================================================

def create_or_load_std_data(mean_results, devs, N, steps, samples, tesselation_func, std_base_dir, noise_type, theta=None):
    """
    Create or load standard deviation data from mean probability distributions.
    
    Args:
        mean_results: List of mean probability distributions for each parameter
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        tesselation_func: Function to create tesselation (dummy for static noise)
        std_base_dir: Base directory for standard deviation data
        noise_type: Type of noise ("static_noise")
    
    Returns:
        List of standard deviation arrays for each deviation value
    """
    import os
    import pickle
    import numpy as np
    
    # Import functions from jaime_scripts and smart_loading_static
    from jaime_scripts import (
        prob_distributions2std
    )
    from smart_loading_static import get_experiment_dir
    
    print(f"\n[DATA] Managing standard deviation data in '{std_base_dir}'...")
    
    # Create base directory for std data
    os.makedirs(std_base_dir, exist_ok=True)
    
    stds = []
    domain = np.arange(N) - N//2  # Center domain around 0
    
    for i, dev in enumerate(devs):
        # Handle new deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # New format: (max_dev, min_factor) or legacy (min, max)
            if dev[1] <= 1.0 and dev[1] >= 0.0:
                # New format
                max_dev, min_factor = dev
                has_noise = max_dev > 0
                dev_str = f"max{max_dev:.3f}_min{max_dev * min_factor:.3f}"
            else:
                # Legacy format
                min_val, max_val = dev
                has_noise = max_val > 0
                dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            # Single value format
            has_noise = dev > 0
            dev_str = f"{dev:.3f}"
        
        # Setup std data directory structure for static noise
        # With unified structure, we always include noise_params (including 0 for no noise)
        noise_params = [dev]  # Static noise uses single parameter
        std_dir = get_experiment_dir(tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type=noise_type, 
                                   base_dir=std_base_dir, theta=theta, samples=samples)
        os.makedirs(std_dir, exist_ok=True)
        
        std_filepath = os.path.join(std_dir, "std_vs_time.pkl")
        
        # Try to load existing std data
        if os.path.exists(std_filepath):
            try:
                with open(std_filepath, 'rb') as f:
                    std_values = pickle.load(f)
                print(f"  [OK] Loaded std data for dev {dev_str}")
                stds.append(std_values)
                continue
            except Exception as e:
                print(f"  [WARNING] Could not load std data for dev {dev_str}: {e}")
        
        # Compute std data from mean probability distributions
        print(f"  [COMPUTING] Computing std data for dev {dev_str}...")
        try:
            # Get mean probability distributions for this deviation
            if mean_results and i < len(mean_results) and mean_results[i]:
                dev_mean_prob_dists = mean_results[i]
                
                # Calculate standard deviations
                if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
                    std_values = prob_distributions2std(dev_mean_prob_dists, domain)
                    
                    # Save std data
                    with open(std_filepath, 'wb') as f:
                        pickle.dump(std_values, f)
                    
                    stds.append(std_values)
                    print(f"  [OK] Computed and saved std data for dev {dev_str} (final std = {std_values[-1]:.3f})")
                else:
                    print(f"  [ERROR] No valid probability distributions found for dev {dev_str}")
                    stds.append([])
            else:
                print(f"  [ERROR] No mean results available for dev {dev_str}")
                stds.append([])
                
        except Exception as e:
            print(f"  [ERROR] Error computing std data for dev {dev_str}: {e}")
            stds.append([])
    
    print(f"[OK] Standard deviation data management completed!")
    return stds

def create_single_archive(archive_args):
    """Worker function to create a single archive in a separate process"""
    root_path, archive_path, temp_archive_name = archive_args
    
    try:
        # Create individual archive
        with tarfile.open(temp_archive_name, "w:gz") as tar:
            tar.add(root_path, arcname=archive_path)
        
        # Get archive size
        archive_size = os.path.getsize(temp_archive_name)
        size_mb = archive_size / (1024 * 1024)
        
        return {
            "success": True,
            "archive_name": temp_archive_name,
            "archive_path": archive_path,
            "size_mb": size_mb,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "archive_name": temp_archive_name,
            "archive_path": archive_path,
            "size_mb": 0,
            "error": str(e)
        }

def create_experiment_archive(N, samples, use_multiprocess=True, max_archive_processes=5, exclude_samples=False, logger=None):
    """
    Create a tar archive of experiment data folders for the specific N value.
    Now supports multiprocess archiving for faster compression of large datasets.
    Includes both samples and probability distribution folders.
    
    Args:
        N: System size
        samples: Number of samples per deviation
        use_multiprocess: Whether to use multiprocess archiving
        max_archive_processes: Maximum number of processes for archiving
        exclude_samples: If True, exclude raw sample files from archive (keep only probDist and std)
        logger: Optional logger for logging archive operations
    """
    def log_and_print(message, level="info"):
        """Helper function to log messages (cluster-safe, no print)"""
        if logger:
            if level == "info":
                logger.info(message.replace("[ARCHIVE] ", "").replace("[WARNING] ", "").replace("[OK] ", "").replace("[ERROR] ", "").replace("[DEBUG] ", "").replace("[INFO] ", ""))
            elif level == "warning":
                logger.warning(message.replace("[WARNING] ", "").replace("[ARCHIVE] ", ""))
            elif level == "error":
                logger.error(message.replace("[ERROR] ", "").replace("[ARCHIVE] ", ""))
    
    try:
        log_and_print("\n[ARCHIVE] Creating tar archive of experiment data...")
        
        # Create archive filename with timestamp, N, and samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_archive_name = f"experiments_data_N{N}_samples{samples}_{timestamp}.tar.gz"
        
        # Define data folders to check based on exclude_samples flag
        if exclude_samples:
            log_and_print("[ARCHIVE] Excluding raw sample files from archive (keeping only processed data)")
            data_folders = [
                "experiments_data_samples_probDist",  # Mean probability distributions
                "experiments_data_samples_std"        # Standard deviation data
            ]
        else:
            data_folders = [
                "experiments_data_samples",           # Raw sample data
                "experiments_data_samples_probDist",  # Mean probability distributions
                "experiments_data_samples_std"        # Standard deviation data
            ]
        
        n_folder_name = f"N_{N}"
        all_folders_to_archive = []
        
        # Find all folders containing N_{N} folders
        for data_folder in data_folders:
            if not os.path.exists(data_folder):
                log_and_print(f"[INFO] Data folder '{data_folder}' not found - skipping")
                continue
                
            log_and_print(f"[ARCHIVE] Checking folder: {os.path.abspath(data_folder)}")
            log_and_print(f"[ARCHIVE] Looking for folders containing '{n_folder_name}'...")
            
            folder_found = False
            for root, dirs, files in os.walk(data_folder):
                if n_folder_name in dirs:
                    # Get the relative path from data folder
                    relative_root = os.path.relpath(root, data_folder)
                    if relative_root == ".":
                        folder_path = n_folder_name
                    else:
                        folder_path = os.path.join(relative_root, n_folder_name)
                    
                    full_path = os.path.join(data_folder, folder_path)
                    archive_path = os.path.join(data_folder, folder_path)  # Keep the full structure
                    all_folders_to_archive.append((full_path, archive_path))
                    log_and_print(f"  Found: {archive_path}")
                    folder_found = True
            
            if not folder_found:
                log_and_print(f"[INFO] No '{n_folder_name}' folders found in {data_folder}")
        
        if not all_folders_to_archive:
            log_and_print(f"[WARNING] No folders found containing '{n_folder_name}' in any data directory - skipping archive creation", "warning")
            log_and_print("[INFO] This is normal if running on a different machine than where computation occurred")
            return None
        
        total_folders = len(all_folders_to_archive)
        log_and_print(f"[ARCHIVE] Found {total_folders} folders to archive across all data directories")
        
        # Determine if we should use multiprocessing
        if not use_multiprocess or total_folders <= 1:
            log_and_print(f"[ARCHIVE] Using single-process archiving (folders: {total_folders})")
            
            # Create the tar archive with all folders
            with tarfile.open(final_archive_name, "w:gz") as tar:
                for i, (full_path, archive_path) in enumerate(all_folders_to_archive, 1):
                    log_and_print(f"  [{i}/{total_folders}] Adding to archive: {archive_path}")
                    tar.add(full_path, arcname=archive_path)
            
        else:
            # Multiprocess archiving approach
            if max_archive_processes is None:
                max_archive_processes = min(total_folders, mp.cpu_count())
            
            log_and_print(f"[ARCHIVE] Using multiprocess archiving with {max_archive_processes} processes")
            log_and_print(f"[ARCHIVE] Creating {total_folders} temporary archives...")
            
            # Prepare arguments for multiprocessing
            temp_archives = []
            archive_args = []
            
            for i, (full_path, archive_path) in enumerate(all_folders_to_archive):
                temp_archive_name = f"temp_archive_N{N}_{i}_{timestamp}.tar.gz"
                temp_archives.append(temp_archive_name)
                archive_args.append((full_path, archive_path, temp_archive_name))
            
            # Create individual archives in parallel
            successful_archives = []
            failed_archives = []
            
            try:
                with ProcessPoolExecutor(max_workers=max_archive_processes) as executor:
                    # Submit all archiving jobs
                    future_to_args = {}
                    for args in archive_args:
                        future = executor.submit(create_single_archive, args)
                        future_to_args[future] = args
                    
                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_args):
                        completed += 1
                        args = future_to_args[future]
                        try:
                            result = future.result()
                            if result["success"]:
                                successful_archives.append(result)
                                log_and_print(f"  [{completed}/{total_folders}] [OK] Created {result['archive_name']} ({result['size_mb']:.1f} MB)")
                            else:
                                failed_archives.append(result)
                                log_and_print(f"  [{completed}/{total_folders}] [FAILED] Failed {result['archive_name']}: {result['error']}", "error")
                                
                        except Exception as e:
                            failed_archives.append({
                                "success": False,
                                "archive_name": args[2],
                                "archive_path": args[1],
                                "size_mb": 0,
                                "error": str(e)
                            })
                            log_and_print(f"  [{completed}/{total_folders}] [EXCEPTION] Exception creating {args[2]}: {e}", "error")
            
            except Exception as e:
                log_and_print(f"[ERROR] Critical error in multiprocess archiving: {e}", "error")
                # Clean up any temp files that were created
                for temp_name in temp_archives:
                    try:
                        if os.path.exists(temp_name):
                            os.remove(temp_name)
                    except:
                        pass
                raise
            
            log_and_print(f"[ARCHIVE] Multiprocess archiving completed: {len(successful_archives)} successful, {len(failed_archives)} failed")
            
            if len(successful_archives) == 0:
                log_and_print("[ERROR] No archives were created successfully", "error")
                return None
            
            # Combine all successful archives into final archive
            log_and_print(f"[ARCHIVE] Combining {len(successful_archives)} archives into final archive: {final_archive_name}")
            
            try:
                with tarfile.open(final_archive_name, "w:gz") as final_tar:
                    for i, result in enumerate(successful_archives, 1):
                        temp_archive_name = result["archive_name"]
                        log_and_print(f"  [{i}/{len(successful_archives)}] Merging {temp_archive_name}")
                        
                        # Extract and re-add contents from temp archive
                        with tarfile.open(temp_archive_name, "r:gz") as temp_tar:
                            for member in temp_tar.getmembers():
                                fileobj = temp_tar.extractfile(member)
                                if fileobj:
                                    final_tar.addfile(member, fileobj)
                                else:
                                    # Handle directories
                                    final_tar.addfile(member)
                
                # Clean up temporary archives
                log_and_print("[ARCHIVE] Cleaning up temporary archives...")
                for result in successful_archives:
                    try:
                        os.remove(result["archive_name"])
                    except Exception as e:
                        log_and_print(f"  Warning: Could not remove {result['archive_name']}: {e}", "warning")
                
                # Also clean up failed temp archives
                for result in failed_archives:
                    try:
                        if os.path.exists(result["archive_name"]):
                            os.remove(result["archive_name"])
                    except:
                        pass
                        
            except Exception as e:
                log_and_print(f"[ERROR] Failed to combine archives: {e}", "error")
                # Clean up temp files
                for result in successful_archives:
                    try:
                        if os.path.exists(result["archive_name"]):
                            os.remove(result["archive_name"])
                    except:
                        pass
                raise
        
        # Get final archive size and report success
        if os.path.exists(final_archive_name):
            archive_size = os.path.getsize(final_archive_name)
            size_mb = archive_size / (1024 * 1024)
            
            log_and_print(f"[OK] Archive created: {final_archive_name} ({size_mb:.1f} MB)")
            log_and_print(f"[OK] Archived {total_folders} folders containing N={N} data")
            log_and_print(f"[OK] Included: samples, probability distributions, and standard deviation data")
            log_and_print(f"[OK] Archive location: {os.path.abspath(final_archive_name)}")
            
            return final_archive_name
        else:
            log_and_print("[ERROR] Final archive was not created", "error")
            return None
        
    except Exception as e:
        log_and_print(f"[ERROR] Failed to create archive: {e}", "error")
        if logger:
            logger.error(traceback.format_exc())
        else:
            traceback.print_exc()
        return None

def create_mean_probability_distributions_multiprocess(
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
    Create mean probability distributions using multiprocessing for parallel computation.
    Each deviation is processed in a separate process for maximum efficiency.
    
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
                logger.info(message.replace("[MEAN_PROB] ", "").replace("[WARNING] ", "").replace("[OK] ", "").replace("[ERROR] ", ""))
            elif level == "warning":
                logger.warning(message.replace("[WARNING] ", "").replace("[MEAN_PROB] ", ""))
            elif level == "error":
                logger.error(message.replace("[ERROR] ", "").replace("[MEAN_PROB] ", ""))
    
    log_and_print("\n[MEAN_PROB] Creating mean probability distributions...")
    start_time = time.time()
    
    # Check if multiprocessing should be used
    if not use_multiprocess or len(devs) <= 1:
        log_and_print(f"[MEAN_PROB] Using single-process mode for {len(devs)} deviations")
        
        # Fall back to original sequential method
        from smart_loading_static import create_mean_probability_distributions
        create_mean_probability_distributions(
            tesselation_func, N, steps, devs, samples,
            source_base_dir, target_base_dir, noise_type, theta
        )
        
        total_time = time.time() - start_time
        log_and_print(f"[OK] Mean probability distributions created in {total_time:.1f}s (sequential)")
        return []
    
    # Multiprocessing approach
    if max_processes is None:
        max_processes = min(len(devs), mp.cpu_count())
    
    log_and_print(f"[MEAN_PROB] Using multiprocess mode with {max_processes} processes for {len(devs)} deviations")
    
    # Prepare arguments for each process
    process_args = []
    for process_id, dev in enumerate(devs):
        args = (dev, process_id, N, steps, samples, source_base_dir, target_base_dir, noise_type, theta, tesselation_func)
        process_args.append(args)
    
    # Track process information
    process_info = {}
    for i, (dev, process_id, *_) in enumerate(process_args):
        # Format dev for filename
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            if dev[1] <= 1.0 and dev[1] >= 0.0:
                max_dev, min_factor = dev
                min_dev = max_dev * min_factor
                dev_str = f"max{max_dev:.3f}_min{min_dev:.3f}"
            else:
                min_val, max_val = dev
                dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            dev_str = f"{float(dev):.3f}"
        
        log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_meanprob_{dev_str}_pid_{process_id}.log")
        process_info[dev] = {
            "process_id": process_id,
            "log_file": log_file,
            "start_time": None,
            "end_time": None,
            "status": "pending"
        }
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=max_processes) as executor:
            # Submit all jobs
            future_to_dev = {}
            for args in process_args:
                dev = args[0]
                future = executor.submit(compute_mean_probability_for_dev, args)
                future_to_dev[future] = dev
                process_info[dev]["start_time"] = time.time()
                process_info[dev]["status"] = "running"
                
                # Format dev for display
                if isinstance(dev, tuple):
                    dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                else:
                    dev_str = f"{dev:.4f}"
                log_and_print(f"[MEAN_PROB] Process launched for dev={dev_str}")
            
            # Collect results as they complete with timeout
            completed = 0
            timeout_start = time.time()
            last_progress_time = timeout_start
            log_and_print(f"[MEAN_PROB] Waiting for {len(future_to_dev)} processes with {MEAN_PROB_TIMEOUT/3600:.1f}h timeout...")
            log_system_resources(logger, "[MEAN_PROB]")
            
            try:
                for future in as_completed(future_to_dev, timeout=MEAN_PROB_TIMEOUT):
                    # Check for shutdown signal
                    if SHUTDOWN_REQUESTED:
                        log_and_print(f"[MEAN_PROB] [SHUTDOWN] Graceful shutdown requested, cancelling remaining processes...", "warning")
                        for remaining_future in future_to_dev:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    
                    dev = future_to_dev[future]
                    completed += 1
                    elapsed = time.time() - timeout_start
                    
                    # Log progress every 5 minutes or on completion
                    if elapsed - (last_progress_time - timeout_start) >= 300 or completed == len(future_to_dev):
                        log_progress_update("MEAN_PROB", completed, len(future_to_dev), timeout_start, logger)
                        log_system_resources(logger, "[MEAN_PROB]")
                        last_progress_time = time.time()
                    
                    try:
                        result = future.result()
                        process_results.append(result)
                        process_info[dev]["end_time"] = time.time()
                        process_info[dev]["status"] = "completed" if result["success"] else "failed"
                        
                        # Format dev for display
                        if isinstance(dev, tuple):
                            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                        else:
                            dev_str = f"{dev:.4f}"
                        
                        if result["success"]:
                            log_and_print(f"[MEAN_PROB] [{completed}/{len(devs)}] [OK] dev={dev_str}: {result['processed_steps']}/{result['total_steps']} steps in {result['total_time']:.1f}s (elapsed: {elapsed/60:.1f}m)")
                        else:
                            log_and_print(f"[MEAN_PROB] [{completed}/{len(devs)}] [FAILED] dev={dev_str}: FAILED - {result['error']}", "error")
                        
                    except Exception as e:
                        process_info[dev]["end_time"] = time.time()
                        process_info[dev]["status"] = "failed"
                        
                        if isinstance(dev, tuple):
                            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                        else:
                            dev_str = f"{dev:.4f}"
                        
                        error_msg = f"Exception in mean probability process for dev {dev_str}: {str(e)}"
                        log_and_print(f"[MEAN_PROB] [{completed}/{len(devs)}] [EXCEPTION] {error_msg} (elapsed: {elapsed/60:.1f}m)", "error")
                        
                        result = {
                            "dev": dev,
                            "process_id": process_info[dev]["process_id"],
                            "processed_steps": 0,
                            "total_steps": steps,
                            "total_time": 0,
                            "log_file": process_info[dev]["log_file"],
                            "success": False,
                            "error": error_msg
                        }
                        process_results.append(result)
            
            except TimeoutError:
                # Handle timeout - try to get partial results
                log_and_print(f"[MEAN_PROB] [TIMEOUT] Mean probability computation timed out after {MEAN_PROB_TIMEOUT/3600:.1f} hours", "error")
                log_and_print(f"[MEAN_PROB] [TIMEOUT] Completed {completed}/{len(future_to_dev)} processes before timeout", "error")
                
                # Cancel remaining futures and collect partial results
                for future in future_to_dev:
                    if not future.done():
                        future.cancel()
                        dev = future_to_dev[future]
                        process_info[dev]["end_time"] = time.time()
                        process_info[dev]["status"] = "timeout"
                        
                        if isinstance(dev, tuple):
                            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                        else:
                            dev_str = f"{dev:.4f}"
                        
                        result = {
                            "dev": dev,
                            "process_id": process_info[dev]["process_id"],
                            "processed_steps": 0,
                            "total_steps": steps,
                            "total_time": 0,
                            "log_file": process_info[dev]["log_file"],
                            "success": False,
                            "error": f"Process timed out after {MEAN_PROB_TIMEOUT/3600:.1f} hours"
                        }
                        process_results.append(result)
                
                # Don't raise exception - let partial results be processed
                log_and_print(f"[MEAN_PROB] [RECOVERY] Continuing with {completed} completed processes", "warning")
    
    except Exception as e:
        log_and_print(f"[ERROR] Critical error in mean probability multiprocessing: {str(e)}", "error")
        if logger:
            logger.error(traceback.format_exc())
        raise
    
    total_time = time.time() - start_time
    
    # Log final results
    successful_processes = sum(1 for r in process_results if r["success"])
    failed_processes = len(process_results) - successful_processes
    
    log_and_print(f"[OK] Mean probability distributions multiprocessing completed in {total_time:.1f}s")
    log_and_print(f"[OK] Results: {successful_processes} successful, {failed_processes} failed processes")
    
    if logger:
        logger.info("MEAN PROBABILITY DISTRIBUTIONS PROCESS SUMMARY:")
        for result in process_results:
            dev = result["dev"]
            if isinstance(dev, tuple):
                dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            
            if result["success"]:
                logger.info(f"  [OK] dev={dev_str}: {result['processed_steps']}/{result['total_steps']} steps in {result['total_time']:.1f}s")
            else:
                logger.error(f"  [FAILED] dev={dev_str}: FAILED - {result['error']}")
    
    return process_results

# @crash_safe_log(log_file_prefix="static_noise_experiment", heartbeat_interval=30.0)
def run_static_experiment():
    """Run the static noise quantum walk experiment with configurable execution modes."""
    
    # Validate configuration
    if CALCULATE_SAMPLES_ONLY and SKIP_SAMPLE_COMPUTATION:
        raise ValueError("Invalid configuration: Cannot set both CALCULATE_SAMPLES_ONLY=True and SKIP_SAMPLE_COMPUTATION=True")
    
    if SKIP_SAMPLE_COMPUTATION and not ENABLE_PLOTTING:
        print("WARNING: Analysis-only mode with plotting disabled - limited output will be generated")
    
    # Configuration summary
    print("=== EXECUTION CONFIGURATION ===")
    mode_description = (
        "Samples Only" if CALCULATE_SAMPLES_ONLY else
        "Analysis Only" if SKIP_SAMPLE_COMPUTATION else
        "Full Pipeline"
    )
    print(f"Execution mode: {mode_description}")
    print(f"Sample computation: {'Enabled' if not SKIP_SAMPLE_COMPUTATION else 'Disabled'}")
    print(f"Analysis phase: {'Enabled' if not CALCULATE_SAMPLES_ONLY else 'Disabled'}")
    print(f"Mean probability multiprocessing: {'Enabled' if USE_MULTIPROCESS_MEAN_PROB else 'Disabled'}")
    if USE_MULTIPROCESS_MEAN_PROB:
        mean_prob_processes = MAX_MEAN_PROB_PROCESSES or "auto-detect"
        print(f"Max mean probability processes: {mean_prob_processes}")
    print(f"Plotting: {'Enabled' if ENABLE_PLOTTING else 'Disabled'}")
    print(f"Archiving: {'Enabled' if CREATE_TAR_ARCHIVE else 'Disabled'}")
    if CREATE_TAR_ARCHIVE:
        print(f"Multiprocess archiving: {'Enabled' if USE_MULTIPROCESS_ARCHIVING else 'Disabled'}")
        if USE_MULTIPROCESS_ARCHIVING:
            archive_processes = MAX_ARCHIVE_PROCESSES or "auto-detect"
            print(f"Max archive processes: {archive_processes}")
        print(f"Exclude samples from archive: {'Yes (processed data only)' if EXCLUDE_SAMPLES_FROM_ARCHIVE else 'No (include all data)'}")
    print(f"Background execution: {'Enabled' if RUN_IN_BACKGROUND else 'Disabled'}")
    print("=" * 40)
    
    # Import required modules at the top
    import numpy as np
    import networkx as nx
    import pickle
    
    # SAFE Background execution handling
    if RUN_IN_BACKGROUND and not os.environ.get('IS_BACKGROUND_PROCESS'):
        print("Starting SAFE background execution...")
        
        try:
            script_path = os.path.abspath(__file__)
            # Use sys.executable to get the current Python interpreter path
            python_executable = sys.executable
            
            # Create environment for subprocess that prevents recursion
            env = os.environ.copy()
            env['IS_BACKGROUND_PROCESS'] = '1'  # This prevents infinite recursion
            
            # Create log and PID file paths
            log_file_path = os.path.join(os.getcwd(), BACKGROUND_LOG_FILE)
            pid_file_path = os.path.join(os.getcwd(), BACKGROUND_PID_FILE)
            
            # Check if there's already a background process running
            if os.path.exists(pid_file_path):
                try:
                    with open(pid_file_path, 'r') as f:
                        old_pid = int(f.read().strip())
                    
                    # Check if the old process is still running
                    if os.name == 'nt':  # Windows
                        result = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}"], 
                                              capture_output=True, text=True)
                        if str(old_pid) in result.stdout:
                            print(f"Background process already running (PID: {old_pid})")
                            print(f"   Kill it first with: taskkill /F /PID {old_pid}")
                            return
                    else:  # Unix-like
                        try:
                            os.kill(old_pid, 0)  # Check if process exists
                            print(f"Background process already running (PID: {old_pid})")
                            print(f"   Kill it first with: kill {old_pid}")
                            return
                        except OSError:
                            pass  # Process doesn't exist, continue
                            
                except (ValueError, OSError):
                    pass  # Invalid PID file, continue
            
            print("Starting background process...")
            
            # Initialize log file
            with open(log_file_path, 'w') as log_file:
                log_file.write(f"Background execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Command: {python_executable} {script_path}\n")
                log_file.write("=" * 50 + "\n\n")
            
            if os.name == 'nt':  # Windows - SAFE METHOD
                # Use subprocess.Popen with proper flags to avoid process spam
                with open(log_file_path, 'a') as log_file:
                    process = subprocess.Popen(
                        [python_executable, "-u", script_path],  # -u for unbuffered output
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW  # Don't create visible window
                    )
                
                # Save PID for cleanup
                with open(pid_file_path, 'w') as pid_file:
                    pid_file.write(str(process.pid))
                
                # Give the process a moment to start and check if it's still running
                time.sleep(0.5)
                if process.poll() is None:
                    print(f"Background process started safely (PID: {process.pid})")
                else:
                    print(f"Warning: Background process (PID: {process.pid}) may have exited immediately")
                    print(f"Check log file for details: {log_file_path}")
                
            else:  # Unix-like systems - SAFE METHOD
                # Use nohup for proper background execution
                with open(log_file_path, 'a') as log_file:
                    # Try different approaches for Unix-like systems
                    process = None
                    
                    # First try with nohup and full detachment
                    try:
                        process = subprocess.Popen(
                            ["nohup", python_executable, "-u", script_path],
                            env=env,
                            cwd=os.getcwd(),
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid,  # Create new session
                            start_new_session=True  # Additional detachment on Python 3.7+
                        )
                    except (TypeError, AttributeError, OSError) as e:
                        print(f"   First attempt failed: {e}")
                        # Fallback 1: Try without start_new_session
                        try:
                            process = subprocess.Popen(
                                ["nohup", python_executable, "-u", script_path],
                                env=env,
                                cwd=os.getcwd(),
                                stdout=log_file,
                                stderr=subprocess.STDOUT,
                                preexec_fn=os.setsid  # Create new session
                            )
                        except (AttributeError, OSError) as e2:
                            print(f"   Second attempt failed: {e2}")
                            # Fallback 2: Try without preexec_fn
                            try:
                                process = subprocess.Popen(
                                    ["nohup", python_executable, "-u", script_path],
                                    env=env,
                                    cwd=os.getcwd(),
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT
                                )
                            except OSError as e3:
                                print(f"   Third attempt failed: {e3}")
                                # Fallback 3: Try without nohup
                                process = subprocess.Popen(
                                    [python_executable, "-u", script_path],
                                    env=env,
                                    cwd=os.getcwd(),
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT
                                )
                
                # Save PID for cleanup
                with open(pid_file_path, 'w') as pid_file:
                    pid_file.write(str(process.pid))
                
                # Give the process a moment to start and check if it's still running
                time.sleep(0.5)
                if process.poll() is None:
                    print(f"Background process started safely (PID: {process.pid})")
                else:
                    print(f"Warning: Background process (PID: {process.pid}) may have exited immediately")
                    print(f"Check log file for details: {log_file_path}")
            
            print(f"Output logged to: {log_file_path}")
            print(f"Process ID saved to: {pid_file_path}")
            print("\n" + "="*50)
            print("SAFE BACKGROUND PROCESS STARTED")
            if os.name == 'nt':  # Windows
                print("   Monitor with: Get-Content " + BACKGROUND_LOG_FILE + " -Wait")
                print("   Kill with: taskkill /F /PID <pid>")
            else:  # Unix-like (Linux/macOS)
                print("   Monitor with: tail -f " + BACKGROUND_LOG_FILE)
                print("   Kill with: kill <pid>")
            print("="*50)
            
            return  # Exit the foreground process
            
        except Exception as e:
            print(f"Error starting background process: {e}")
            print("   Falling back to foreground execution...")
    
    # Check if we're the background process
    if os.environ.get('IS_BACKGROUND_PROCESS'):
        print("Running in SAFE background mode...")
        print(f"   Process ID: {os.getpid()}")
        print(f"   Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Set up clean signal handlers for graceful shutdown
        def cleanup_and_exit(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            try:
                pid_file_path = os.path.join(os.getcwd(), BACKGROUND_PID_FILE)
                if os.path.exists(pid_file_path):
                    os.remove(pid_file_path)
                    print("Cleaned up PID file")
            except Exception as e:
                print(f"Warning during cleanup: {e}")
            print("Background process exiting cleanly")
            sys.exit(0)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, cleanup_and_exit)
        signal.signal(signal.SIGTERM, cleanup_and_exit)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, cleanup_and_exit)
    
    # Import additional required modules
    try:
        # For static noise, we don't need tesselations module since it's built-in
        from sqw.experiments_expanded_static import running
        from sqw.states import uniform_initial_state, amp2prob
        
        # Import shared functions from jaime_scripts and smart_loading_static
        from jaime_scripts import (
            prob_distributions2std,
            plot_std_vs_time_qwak
        )
        from smart_loading_static import smart_load_or_create_experiment, get_experiment_dir
        
        print("Successfully imported all required modules")
    except ImportError as e:
        error_msg = f"Error: Could not import required modules: {e}"
        print(error_msg)
        print("Make sure you're running this script from the correct directory with all dependencies available")
        
        # If we're in background mode, write the error to the log file
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                with open(BACKGROUND_LOG_FILE, 'a') as f:
                    f.write(f"\n{error_msg}\n")
                    f.write("Script exiting due to import error\n")
            except:
                pass
        
        # Clean up PID file if we're the background process
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                if os.path.exists(BACKGROUND_PID_FILE):
                    os.remove(BACKGROUND_PID_FILE)
            except:
                pass
        
        raise

    print("Starting static noise quantum walk experiment...")
    
    # Memory safety check (updated for streaming approach)
    # With streaming, we only hold one state at a time, not all states
    estimated_memory_per_process_mb = (N * 16 * 3) / (1024 * 1024)  # ~3 states max (current + temp calculations)
    total_estimated_memory_mb = estimated_memory_per_process_mb * MAX_PROCESSES
    
    print(f"Memory estimation (streaming approach):")
    print(f"  Per process: ~{estimated_memory_per_process_mb:.0f} MB (single state + overhead)")
    print(f"  Total (all processes): ~{total_estimated_memory_mb:.0f} MB")
    print(f"  Traditional approach would need: ~{(steps * N * 16 * 2 * MAX_PROCESSES) / (1024 * 1024):.0f} MB")
    
    if total_estimated_memory_mb > 8000:  # This threshold is much less likely to be hit now
        print("[WARNING] High memory usage predicted!")
        print("   Consider reducing N or MAX_PROCESSES")
    else:
        print("[OK] Memory usage looks reasonable with streaming approach")
    
    print(f"Experiment parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Multiprocessing: Using up to {MAX_PROCESSES} processes for {len(devs)} deviations")
    print("MULTIPROCESS MODE: Each deviation will run in a separate process!")
    
    # Setup master logging
    master_logger, master_log_file = setup_master_logging()
    master_logger.info("=" * 60)
    master_logger.info("MULTIPROCESS STATIC NOISE EXPERIMENT STARTED")
    master_logger.info("=" * 60)
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}")
    master_logger.info(f"Deviations: {devs}")
    master_logger.info(f"Max processes: {MAX_PROCESSES}")
    master_logger.info(f"Master log file: {master_log_file}")
    
    # Start timing the main experiment
    start_time = time.time()
    total_samples = len(devs) * samples
    completed_samples = 0

    # Sample computation phase with multiprocessing
    experiment_time = 0
    process_results = []
    
    if not SKIP_SAMPLE_COMPUTATION:
        master_logger.info("=" * 40)
        master_logger.info("MULTIPROCESS SAMPLE COMPUTATION PHASE")
        master_logger.info("=" * 40)
        
        # Prepare arguments for each process
        process_args = []
        for process_id, dev in enumerate(devs):
            args = (dev, process_id, N, steps, samples, theta, initial_state_kwargs)
            process_args.append(args)
        
        master_logger.info(f"Launching {len(process_args)} processes...")
        
        # Track process information
        process_info = {}
        for i, (dev, process_id, *_) in enumerate(process_args):
            # Format dev for filename (handle both old and new formats)
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                # New format: (max_dev, min_factor) or legacy (min, max)
                if dev[1] <= 1.0 and dev[1] >= 0.0:
                    max_dev, min_factor = dev
                    min_dev = max_dev * min_factor
                    dev_str = f"max{max_dev:.3f}_min{min_dev:.3f}"
                else:
                    min_val, max_val = dev
                    dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
            else:
                # Single value format
                dev_str = f"{float(dev):.3f}"
                
            log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_pid_{process_id}.log")
            process_info[dev] = {
                "process_id": process_id,
                "log_file": log_file,
                "start_time": None,
                "end_time": None,
                "status": "pending"
            }
        
        # Execute processes concurrently with robust error handling
        max_retries = 3
        retry_delay = 30  # seconds
        completed_samples = 0
        total_samples = len(devs) * samples
        process_timeout_but_likely_completed = False  # Flag for timeout recovery
        
        for attempt in range(max_retries):
            try:
                master_logger.info(f"Starting process pool attempt {attempt + 1}/{max_retries}")
                
                with ProcessPoolExecutor(max_workers=MAX_PROCESSES) as executor:
                    # Submit all jobs
                    future_to_dev = {}
                    for args in process_args:
                        dev = args[0]
                        future = executor.submit(compute_dev_samples, args)
                        future_to_dev[future] = dev
                        process_info[dev]["start_time"] = time.time()
                        process_info[dev]["status"] = "running"
                        # Format dev for display
                        if isinstance(dev, tuple):
                            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                        else:
                            dev_str = f"{dev:.4f}"
                        master_logger.info(f"Process launched for dev={dev_str} (PID will be assigned)")
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_dev, timeout=PROCESS_TIMEOUT):  # Dynamic timeout based on problem size
                        # Check for shutdown signal
                        if SHUTDOWN_REQUESTED:
                            master_logger.warning("[SAMPLES] [SHUTDOWN] Graceful shutdown requested, cancelling remaining processes...")
                            for remaining_future in future_to_dev:
                                if not remaining_future.done():
                                    remaining_future.cancel()
                            break
                        
                        dev = future_to_dev[future]
                        try:
                            result = future.result()
                            process_results.append(result)
                            process_info[dev]["end_time"] = time.time()
                            
                            if result["success"]:
                                process_info[dev]["status"] = "completed"
                                completed_samples += result["computed_samples"]
                                # Format dev for display
                                if isinstance(dev, tuple):
                                    dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                                else:
                                    dev_str = f"{dev:.4f}"
                                master_logger.info(f"[OK] Process for dev={dev_str} completed successfully")
                                master_logger.info(f"  Computed samples: {result['computed_samples']}")
                                master_logger.info(f"  Time: {result['total_time']:.1f}s")
                                master_logger.info(f"  Log file: {result['log_file']}")
                            else:
                                process_info[dev]["status"] = "failed"
                                # Format dev for display
                                if isinstance(dev, tuple):
                                    dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                                else:
                                    dev_str = f"{dev:.4f}"
                                master_logger.error(f"[FAILED] Process for dev={dev_str} failed")
                                master_logger.error(f"  Error: {result['error']}")
                                master_logger.error(f"  Log file: {result['log_file']}")
                                
                        except Exception as e:
                            import traceback
                            process_info[dev]["status"] = "failed"
                            process_info[dev]["end_time"] = time.time()
                            # Format dev for display
                            if isinstance(dev, tuple):
                                dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
                            else:
                                dev_str = f"{dev:.4f}"
                            error_msg = f"Exception in process for dev={dev_str}: {str(e)}"
                            master_logger.error(error_msg)
                            master_logger.error(traceback.format_exc())
                            
                            process_results.append({
                                "dev": dev,
                                "process_id": -1,
                                "computed_samples": 0,
                                "total_time": 0,
                                "log_file": "unknown",
                                "success": False,
                                "error": error_msg
                            })
                
                # If we get here, all processes completed (successfully or not)
                master_logger.info(f"Process pool attempt {attempt + 1} completed")
                break
                
            except TimeoutError as te:
                # Check if some processes might have completed but not returned
                unfinished_count = sum(1 for dev in devs if process_info[dev]["status"] == "pending")
                master_logger.error(f"Process pool attempt {attempt + 1} failed: {unfinished_count} (of {len(devs)}) futures unfinished")
                master_logger.error(f"Timeout of {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours) exceeded")
                
                # Check process logs to see if processes actually completed
                completed_in_logs = 0
                for dev in devs:
                    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
                    log_file = os.path.join(PROCESS_LOG_DIR, f"process_dev_{dev_str}_pid_*.log")
                    import glob
                    matching_logs = glob.glob(log_file)
                    if matching_logs:
                        try:
                            with open(matching_logs[0], 'r') as f:
                                content = f.read()
                                if "completed:" in content:
                                    completed_in_logs += 1
                                    master_logger.info(f"  Process for dev={dev_str} appears to have completed in log file")
                        except:
                            pass
                
                if completed_in_logs > 0:
                    master_logger.error(f"IMPORTANT: {completed_in_logs} processes may have completed but timed out in coordination")
                    master_logger.error("  Check individual process logs and output directories for completed data")
                    master_logger.error("  Consider running with SKIP_SAMPLE_COMPUTATION=True to analyze existing data")
                    
                    # If most/all processes completed in logs, we might be able to continue
                    if completed_in_logs >= len(devs) * 0.8:  # 80% or more completed
                        master_logger.info(f"RECOVERY: {completed_in_logs}/{len(devs)} processes appear completed in logs")
                        master_logger.info("RECOVERY: Will attempt to continue with analysis phase")
                        # Set a flag to continue despite timeout
                        process_timeout_but_likely_completed = True
                        break  # Exit retry loop and continue
                
                if attempt < max_retries - 1:
                    master_logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Reset process info for retry
                    for dev in process_info:
                        process_info[dev]["status"] = "pending"
                        process_info[dev]["start_time"] = None
                        process_info[dev]["end_time"] = None
                    process_results.clear()
                    completed_samples = 0
                else:
                    if process_timeout_but_likely_completed:
                        master_logger.warning("Timeout occurred but processes likely completed - continuing with analysis")
                        break  # Exit retry loop and continue with analysis
                    else:
                        master_logger.error("All retry attempts failed")
                        raise
                
            except Exception as e:
                import traceback
                master_logger.error(f"Process pool attempt {attempt + 1} failed: {str(e)}")
                master_logger.error(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    master_logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    # Reset process info for retry
                    for dev in process_info:
                        process_info[dev]["status"] = "pending"
                        process_info[dev]["start_time"] = None
                        process_info[dev]["end_time"] = None
                    process_results.clear()
                    completed_samples = 0
                else:
                    if process_timeout_but_likely_completed:
                        master_logger.warning("Timeout occurred but processes likely completed - continuing with analysis")
                        break  # Exit retry loop and continue with analysis
                    else:
                        master_logger.error("All retry attempts failed") 
                        raise

        experiment_time = time.time() - start_time
        
        # Log final results
        master_logger.info("=" * 40)
        master_logger.info("MULTIPROCESS COMPUTATION COMPLETED")
        master_logger.info("=" * 40)
        master_logger.info(f"Total execution time: {experiment_time:.2f} seconds")
        master_logger.info(f"Total samples computed: {completed_samples}/{total_samples}")
        
        # Log individual process results
        master_logger.info("\nPROCESS SUMMARY:")
        successful_processes = 0
        failed_processes = 0
        
        for result in process_results:
            dev = result["dev"]
            # Format dev for display
            if isinstance(dev, tuple):
                dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            
            if result["success"]:
                successful_processes += 1
                master_logger.info(f"  [OK] dev={dev_str}: {result['computed_samples']} samples in {result['total_time']:.1f}s")
            else:
                failed_processes += 1
                master_logger.error(f"  [FAILED] dev={dev_str}: FAILED - {result['error']}")
        
        master_logger.info(f"\nRESULTS: {successful_processes} successful, {failed_processes} failed processes")
        
        # Log process log file locations
        master_logger.info("\nPROCESS LOG FILES:")
        for dev, info in process_info.items():
            # Format dev for display
            if isinstance(dev, tuple):
                dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
            else:
                dev_str = f"{dev:.4f}"
            master_logger.info(f"  dev={dev_str}: {info['log_file']}")
        
        print(f"\n[COMPLETED] Multiprocess sample computation completed in {experiment_time:.2f} seconds")
        print(f"Total samples computed: {completed_samples}")
        print(f"Successful processes: {successful_processes}/{len(devs)}")
        print(f"Master log file: {master_log_file}")
        print(f"Process log directory: {PROCESS_LOG_DIR}")
        
    else:
        master_logger.info("SKIPPING SAMPLE COMPUTATION")
        print("=== SKIPPING SAMPLE COMPUTATION ===")
        print("Sample computation disabled - proceeding to analysis phase")
        experiment_time = 0
        completed_samples = 0

    # Early exit if only computing samples
    if CALCULATE_SAMPLES_ONLY:
        master_logger.info("SAMPLES ONLY MODE - ANALYSIS SKIPPED")
        print("\n=== SAMPLES ONLY MODE - ANALYSIS SKIPPED ===")
        print("Sample computation completed. Skipping analysis and plotting.")
        
        # Create tar archive if enabled (even in samples-only mode)
        if CREATE_TAR_ARCHIVE:
            master_logger.info("Creating tar archive...")
            archive_name = create_experiment_archive(N, samples, USE_MULTIPROCESS_ARCHIVING, MAX_ARCHIVE_PROCESSES, EXCLUDE_SAMPLES_FROM_ARCHIVE, master_logger)
            if archive_name:
                master_logger.info(f"Archive created successfully: {archive_name}")
            else:
                master_logger.warning("Archive creation failed or was skipped")
        else:
            master_logger.info("Archiving disabled")
            print("Archiving disabled (CREATE_TAR_ARCHIVE=False)")
        
        print("To run analysis on existing samples, set:")
        print("  CALCULATE_SAMPLES_ONLY = False")
        print("  SKIP_SAMPLE_COMPUTATION = True")
        
        total_time = time.time() - start_time
        master_logger.info(f"Total execution time: {total_time:.2f} seconds")
        master_logger.info("Experiment completed (samples only mode)")
        
        print(f"Total execution time: {total_time:.2f} seconds")
        
        return {
            "mode": "samples_only",
            "devs": devs,
            "N": N,
            "steps": steps,
            "samples": samples,
            "total_time": total_time,
            "theta": theta,
            "completed_samples": completed_samples,
            "multiprocessing": True,
            "mean_prob_multiprocessing": USE_MULTIPROCESS_MEAN_PROB,
            "max_mean_prob_processes": MAX_MEAN_PROB_PROCESSES,
            "archiving_enabled": CREATE_TAR_ARCHIVE,
            "multiprocess_archiving": USE_MULTIPROCESS_ARCHIVING,
            "max_archive_processes": MAX_ARCHIVE_PROCESSES,
            "process_results": process_results,
            "master_log_file": master_log_file,
            "process_log_dir": PROCESS_LOG_DIR
        }

    # Analysis phase
    print("\n=== ANALYSIS PHASE ===")
    print("Loading existing samples and computing analysis...")

    # Smart load or create mean probability distributions with multiprocessing support
    print("\n[DATA] Smart loading/creating mean probability distributions...")
    master_logger.info("=" * 40)
    master_logger.info("ANALYSIS PHASE - MEAN PROBABILITY DISTRIBUTIONS")
    master_logger.info("=" * 40)
    
    try:
        # First, try to load existing mean probability distributions
        from smart_loading_static import (
            check_mean_probability_distributions_exist, 
            load_mean_probability_distributions
        )
        
        if check_mean_probability_distributions_exist(dummy_tesselation_func, N, steps, devs, samples,
                                                    "experiments_data_samples_probDist", "static_noise", theta):
            print("[OK] Found existing mean probability distributions - loading directly!")
            master_logger.info("Loading existing mean probability distributions")
            mean_results = load_mean_probability_distributions(
                dummy_tesselation_func, N, steps, devs, samples,
                "experiments_data_samples_probDist", "static_noise", theta
            )
            
            # Validate loaded data
            if mean_results is None:
                raise ValueError("Failed to load mean probability distributions - all data appears to be corrupted")
            
            # Check for corrupted data and warn user
            corrupted_count = 0
            total_count = 0
            for dev_idx, dev_data in enumerate(mean_results):
                if dev_data is not None:
                    none_count = sum(1 for step_data in dev_data if step_data is None)
                    corrupted_count += none_count
                    total_count += len(dev_data)
                    if none_count > 0:
                        print(f"    WARNING: Dev {dev_idx+1}/{len(devs)} has {none_count}/{len(dev_data)} corrupted time steps")
            
            if corrupted_count > 0:
                print(f"    TOTAL: {corrupted_count}/{total_count} time steps have corrupted data")
                print(f"    Consider regenerating the corrupted data for better results")
            
            print("[OK] Mean probability distributions loaded successfully")
            master_logger.info("Mean probability distributions loaded successfully")
        else:
            # Need to create mean probability distributions
            print("[INFO] Mean probability distributions not found - creating from samples...")
            master_logger.info("Creating mean probability distributions from samples")
            
            # Check if multiprocessing is enabled and beneficial
            if USE_MULTIPROCESS_MEAN_PROB and len(devs) > 1:
                if MAX_MEAN_PROB_PROCESSES is None:
                    mean_prob_processes = min(len(devs), mp.cpu_count())
                else:
                    mean_prob_processes = MAX_MEAN_PROB_PROCESSES
                
                print(f"[MULTIPROCESS] Creating mean probability distributions using {mean_prob_processes} processes")
                master_logger.info(f"Using multiprocess mean probability calculation with {mean_prob_processes} processes")
                
                # Use multiprocessing approach
                mean_prob_results = create_mean_probability_distributions_multiprocess(
                    tesselation_func=dummy_tesselation_func,
                    N=N,
                    steps=steps,
                    devs=devs,
                    samples=samples,
                    source_base_dir="experiments_data_samples",
                    target_base_dir="experiments_data_samples_probDist",
                    noise_type="static_noise",
                    theta=theta,
                    use_multiprocess=True,
                    max_processes=mean_prob_processes,
                    logger=master_logger
                )
                
                # Log process results
                successful_mean_prob = sum(1 for r in mean_prob_results if r["success"])
                failed_mean_prob = len(mean_prob_results) - successful_mean_prob
                master_logger.info(f"Mean probability multiprocessing: {successful_mean_prob} successful, {failed_mean_prob} failed")
                
                if successful_mean_prob < len(devs):
                    print(f"[WARNING] Some mean probability processes failed ({failed_mean_prob}/{len(devs)})")
                    master_logger.warning(f"Some mean probability processes failed: {failed_mean_prob}/{len(devs)}")
            else:
                print("[INFO] Using sequential mean probability calculation")
                master_logger.info("Using sequential mean probability calculation")
                
                # Use sequential approach
                from smart_loading_static import create_mean_probability_distributions
                create_mean_probability_distributions(
                    dummy_tesselation_func, N, steps, devs, samples,
                    "experiments_data_samples", "experiments_data_samples_probDist", 
                    "static_noise", theta
                )
            
            # Load the created distributions
            print("[DATA] Loading newly created mean probability distributions...")
            mean_results = load_mean_probability_distributions(
                dummy_tesselation_func, N, steps, devs, samples,
                "experiments_data_samples_probDist", "static_noise", theta
            )
            print("[OK] Mean probability distributions created and loaded successfully")
            master_logger.info("Mean probability distributions created and loaded successfully")
        
    except Exception as e:
        error_msg = f"Warning: Could not smart load/create mean probability distributions: {e}"
        print(f"[WARNING] {error_msg}")
        master_logger.error(error_msg)
        import traceback
        master_logger.error(traceback.format_exc())
        mean_results = None

    # Create or load standard deviation data
    try:
        stds = create_or_load_std_data(
            mean_results, devs, N, steps, samples, dummy_tesselation_func,
            "experiments_data_samples_std", "static_noise", theta=theta
        )
        
        # Print final std values for verification
        for i, (dev, std_values) in enumerate(zip(devs, stds)):
            if std_values and len(std_values) > 0:
                # Format dev for display
                if isinstance(dev, tuple) and len(dev) == 2:
                    if dev[1] <= 1.0 and dev[1] >= 0.0:
                        # New format: (max_dev, min_factor)
                        max_dev, min_factor = dev
                        dev_label = f"max{max_dev:.3f}_min{max_dev*min_factor:.3f}"
                    else:
                        # Legacy format: (min, max)
                        min_val, max_val = dev
                        dev_label = f"min{min_val:.3f}_max{max_val:.3f}"
                else:
                    # Single value format
                    dev_label = f"{dev:.3f}"
                
                print(f"Dev {dev_label}: Final std = {std_values[-1]:.3f}")
            else:
                # Format dev for display
                if isinstance(dev, tuple) and len(dev) == 2:
                    if dev[1] <= 1.0 and dev[1] >= 0.0:
                        # New format: (max_dev, min_factor)
                        max_dev, min_factor = dev
                        dev_label = f"max{max_dev:.3f}_min{max_dev*min_factor:.3f}"
                    else:
                        # Legacy format: (min, max)
                        min_val, max_val = dev
                        dev_label = f"min{min_val:.3f}_max{max_val:.3f}"
                else:
                    # Single value format
                    dev_label = f"{dev:.3f}"
                
                print(f"Dev {dev_label}: No valid standard deviation data")
                
    except Exception as e:
        print(f"[WARNING] Warning: Could not create/load standard deviation data: {e}")
        stds = []

    # Plot standard deviation vs time if enabled
    if ENABLE_PLOTTING:
        print("\n[PLOT] Creating standard deviation vs time plot...")
        try:
            if 'stds' in locals() and len(stds) > 0 and any(len(std) > 0 for std in stds):
                import matplotlib.pyplot as plt
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                for i, (std_values, dev) in enumerate(zip(stds, devs)):
                    if len(std_values) > 0:
                        time_steps = list(range(len(std_values)))
                        
                        # Format dev for display
                        if isinstance(dev, tuple) and len(dev) == 2:
                            if dev[1] <= 1.0 and dev[1] >= 0.0:
                                # New format: (max_dev, min_factor)
                                max_dev, min_factor = dev
                                dev_label = f"max{max_dev:.3f}_min{max_dev*min_factor:.3f}"
                            else:
                                # Legacy format: (min, max)
                                min_val, max_val = dev
                                dev_label = f"min{min_val:.3f}_max{max_val:.3f}"
                        else:
                            # Single value format
                            dev_label = f"{dev:.3f}"
                        
                        # Handle zero values for log-log plot
                        if USE_LOGLOG_PLOT:
                            # Check if this is a zero standard deviation case (noiseless)
                            if all(s == 0 for s in std_values):
                                # For noiseless case (std = 0), plot at bottom of y-axis
                                # Use a small epsilon value to make it visible on log scale
                                epsilon = 1e-3  # Small value for visualization
                                filtered_times = [t for t in time_steps if t > 0]
                                filtered_stds = [epsilon] * len(filtered_times)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=f'Static deviation = {dev_label} (noiseless)', 
                                         marker='s', markersize=4, linewidth=2, 
                                         linestyle='--', alpha=0.8)
                            else:
                                # Remove zero values which can't be plotted on log scale
                                filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                                if filtered_data:
                                    filtered_times, filtered_stds = zip(*filtered_data)
                                    plt.loglog(filtered_times, filtered_stds, 
                                             label=f'Static deviation = {dev_label}', 
                                             marker='o', markersize=3, linewidth=2)


                        else:
                            plt.plot(time_steps, std_values, 
                                   label=f'Static deviation = {dev_label}', 
                                   marker='o', markersize=3, linewidth=2)
                
                plt.xlabel('Time Step', fontsize=12)
                plt.ylabel('Standard Deviation', fontsize=12)
                
                if USE_LOGLOG_PLOT:
                    plt.title('Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3, which="both", ls="-")  # Grid for both major and minor ticks
                    plot_filename = "static_noise_std_vs_time_loglog.png"
                else:
                    plt.title('Standard Deviation vs Time for Different Static Noise Deviations', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plot_filename = "static_noise_std_vs_time.png"
                
                plt.legend(fontsize=10)
                plt.tight_layout()
                
                # Save the plot (if enabled)
                if SAVE_FIGURES:
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"[OK] Plot saved as '{plot_filename}'")
                
                # Show the plot
                plt.show()
                plot_type = "log-log" if USE_LOGLOG_PLOT else "linear"
                saved_status = " and saved" if SAVE_FIGURES else ""
                print(f"[OK] Standard deviation plot displayed{saved_status}! (Scale: {plot_type})")
            else:
                print("[WARNING] Warning: No standard deviation data available for plotting")
        except Exception as e:
            print(f"[WARNING] Warning: Could not create plot: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n[PLOT] Plotting disabled (ENABLE_PLOTTING=False)")

    # Plot final probability distributions if enabled
    if ENABLE_PLOTTING and PLOT_FINAL_PROBDIST:
        print("\n[PLOT] Creating final probability distribution plot...")
        try:
            if 'mean_results' in locals() and mean_results and len(mean_results) > 0:
                import matplotlib.pyplot as plt
                
                # Create the plot
                plt.figure(figsize=(14, 8))
                
                # Use the last time step (steps-1)
                final_step = steps - 1
                domain = np.arange(N) - N//2  # Center domain around 0
                
                for i, (dev_mean_prob_dists, dev) in enumerate(zip(mean_results, devs)):
                    if dev_mean_prob_dists and len(dev_mean_prob_dists) > final_step and dev_mean_prob_dists[final_step] is not None:
                        final_prob_dist = dev_mean_prob_dists[final_step].flatten()
                        
                        # Format dev for display
                        if isinstance(dev, tuple) and len(dev) == 2:
                            if dev[1] <= 1.0 and dev[1] >= 0.0:
                                # New format: (max_dev, min_factor)
                                max_dev, min_factor = dev
                                dev_label = f"max{max_dev:.3f}_min{max_dev*min_factor:.3f}"
                            else:
                                # Legacy format: (min, max)
                                min_val, max_val = dev
                                dev_label = f"min{min_val:.3f}_max{max_val:.3f}"
                        else:
                            # Single value format
                            dev_label = f"{dev:.3f}"
                        
                        # Plot the probability distribution with log y-axis
                        plt.semilogy(domain, final_prob_dist, 
                                   label=f'Static deviation = {dev_label}', 
                                   linewidth=2, alpha=0.8)
                
                plt.xlabel('Position', fontsize=12)
                plt.ylabel('Probability (log scale)', fontsize=12)
                plt.title(f'Probability Distributions at Final Time Step (t={final_step}) - Log Scale', fontsize=14)
                plt.xlim(-150, 150)  # Limit x-axis range to -150 to 150
                plt.ylim(1e-20, None)  # Limit y-axis minimum to 10^-20
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Save the plot (if enabled)
                probdist_filename = "static_noise_final_probdist_log.png"
                if SAVE_FIGURES:
                    plt.savefig(probdist_filename, dpi=300, bbox_inches='tight')
                    print(f"[OK] Probability distribution plot saved as '{probdist_filename}'")
                
                # Show the plot
                plt.show()
                saved_status = " and saved" if SAVE_FIGURES else ""
                print(f"[OK] Final probability distribution plot displayed{saved_status}!")
            else:
                print("[WARNING] Warning: No mean probability distribution data available for plotting")
        except Exception as e:
            print(f"[WARNING] Warning: Could not create probability distribution plot: {e}")
            import traceback
            traceback.print_exc()
    elif not ENABLE_PLOTTING:
        print("\n[PLOT] Probability distribution plotting disabled (ENABLE_PLOTTING=False)")
    else:
        print("\n[PLOT] Final probability distribution plotting disabled (PLOT_FINAL_PROBDIST=False)")

    # Create tar archive if enabled
    if CREATE_TAR_ARCHIVE:
        archive_name = create_experiment_archive(N, samples, USE_MULTIPROCESS_ARCHIVING, MAX_ARCHIVE_PROCESSES, EXCLUDE_SAMPLES_FROM_ARCHIVE, master_logger)
        if archive_name:
            print(f"[OK] Archive created successfully: {archive_name}")
        else:
            print("[WARNING] Archive creation failed or was skipped")
    else:
        print("[INFO] Archiving disabled (CREATE_TAR_ARCHIVE=False)")

    print("Static noise experiment completed successfully!")
    total_time = time.time() - start_time
    master_logger.info(f"EXPERIMENT COMPLETED - Total time: {total_time:.2f} seconds")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    print("\n=== Performance Summary ===")
    print(f"Execution mode: {'Samples Only' if CALCULATE_SAMPLES_ONLY else 'Analysis Only' if SKIP_SAMPLE_COMPUTATION else 'Full Pipeline'}")
    print(f"System size (N): {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(devs)}")
    print(f"Multiprocessing: {MAX_PROCESSES} max processes")
    
    if not SKIP_SAMPLE_COMPUTATION:
        print(f"Total quantum walks computed: {completed_samples}")
        successful_processes = len([r for r in process_results if r["success"]])
        print(f"Successful processes: {successful_processes}/{len(devs)}")
        if experiment_time > 0 and completed_samples > 0:
            print(f"Average time per quantum walk: {experiment_time / completed_samples:.3f} seconds")
    else:
        print(f"Expected quantum walks: {len(devs) * samples} (sample computation skipped)")
    
    print("\n=== Multiprocessing Log Files ===")
    print(f"Master log: {master_log_file}")
    print(f"Process logs directory: {PROCESS_LOG_DIR}")
    if process_results:
        print("Individual process logs:")
        for result in process_results:
            status = "[OK]" if result["success"] else "[FAILED]"
            # Format dev properly for both single values and tuples
            dev = result['dev']
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                # New format: (max_dev, min_factor) or legacy (min, max)
                if dev[1] <= 1.0 and dev[1] >= 0.0:
                    max_dev, min_factor = dev
                    min_dev = max_dev * min_factor
                    dev_str = f"max{max_dev:.4f}_min{min_dev:.4f}"
                else:
                    min_val, max_val = dev
                    dev_str = f"min{min_val:.4f}_max{max_val:.4f}"
            else:
                # Single value format
                dev_str = f"{float(dev):.4f}"
            print(f"  {status} dev={dev_str}: {result['log_file']}")

    print("\n=== Execution Modes ===")
    print("Available execution modes:")
    print("1. Full Pipeline (default): Compute samples + analysis + plots + archive")
    print("2. Samples Only: Set CALCULATE_SAMPLES_ONLY = True")
    print("3. Analysis Only: Set SKIP_SAMPLE_COMPUTATION = True")
    print("4. Custom: Adjust individual toggles for plotting, archiving, etc.")
    
    print("\n=== Static Noise Details ===")
    print("Static noise model:")
    print(f"- dev=0: Perfect static evolution with theta={theta:.3f} (no noise)")
    print("- dev>0: Random deviation applied to Hamiltonian edges with range 'dev'")
    print("- Each sample generates different random noise for edge parameters")
    print("- Mean probability distributions average over all samples")
    print("- Tessellations are built-in (alpha and beta patterns)")
    print("- MULTIPROCESSING: Each deviation value runs in separate process")
    
    print("\n=== Plotting Features ===")
    print(f"- Plotting enabled: {ENABLE_PLOTTING}")
    print(f"- Save figures to files: {SAVE_FIGURES}")
    if ENABLE_PLOTTING:
        plot_type = "Log-log scale" if USE_LOGLOG_PLOT else "Linear scale"
        plot_filename = "static_noise_std_vs_time_loglog_N{N}_samples{samples}.png" if USE_LOGLOG_PLOT else "static_noise_std_vs_time.png"
        print(f"- Standard deviation plot type: {plot_type}")
        if SAVE_FIGURES:
            print(f"- Standard deviation plot saved as: {plot_filename}")
        if USE_LOGLOG_PLOT:
            print("- Log-log plots help identify power-law scaling behavior sigma(t) proportional to t^alpha")
        
        print(f"- Final probability distribution plot enabled: {PLOT_FINAL_PROBDIST}")
        if PLOT_FINAL_PROBDIST:
            if SAVE_FIGURES:
                print("- Final probability distribution plot saved as: static_noise_final_probdist_log.png")
            print("- Shows probability distributions at the final time step for all deviations")
            print("- Uses log scale for y-axis and focuses on position range -150 to +150")
    
    print("\n=== Archive Features ===")
    print(f"- Create tar archive: {CREATE_TAR_ARCHIVE}")
    if CREATE_TAR_ARCHIVE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"experiments_data_samples_N{N}_samples{samples}_{timestamp}.tar.gz"
        print(f"- Archive will be saved as: experiments_data_samples_N{N}_samples{samples}_[timestamp].tar.gz")
        print(f"- Archive contains only N={N} folders and their parent directory structure")
        print("- This selective archiving reduces file size compared to archiving all N values")
        print(f"- Multiprocess archiving: {USE_MULTIPROCESS_ARCHIVING}")
        if USE_MULTIPROCESS_ARCHIVING:
            archive_processes = MAX_ARCHIVE_PROCESSES or "auto-detect"
            print(f"- Max archive processes: {archive_processes}")
            print("- Multiprocess archiving creates temporary archives in parallel, then combines them")
            print("- This significantly speeds up archiving of large datasets with many folders")
    
    return {
        "mode": "full_pipeline",
        "devs": devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time,
        "theta": theta,
        "completed_samples": completed_samples,
        "sample_computation_enabled": not SKIP_SAMPLE_COMPUTATION,
        "analysis_enabled": not CALCULATE_SAMPLES_ONLY,
        "plotting_enabled": ENABLE_PLOTTING,
        "mean_prob_multiprocessing": USE_MULTIPROCESS_MEAN_PROB,
        "max_mean_prob_processes": MAX_MEAN_PROB_PROCESSES,
        "archiving_enabled": CREATE_TAR_ARCHIVE,
        "multiprocess_archiving": USE_MULTIPROCESS_ARCHIVING,
        "max_archive_processes": MAX_ARCHIVE_PROCESSES,
        "multiprocessing": True,
        "max_processes": MAX_PROCESSES,
        "process_results": process_results,
        "master_log_file": master_log_file,
        "process_log_dir": PROCESS_LOG_DIR
    }

if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    try:
        run_static_experiment()
    except Exception as e:
        error_msg = f"Fatal error in run_static_experiment: {e}"
        print(error_msg)
        
        # Try to log to master logger if available
        try:
            master_logger = logging.getLogger("master")
            if master_logger.handlers:
                master_logger.error(error_msg)
                import traceback
                master_logger.error(traceback.format_exc())
        except:
            pass
        
        # If we're in background mode, write the error to the log file
        if os.environ.get('IS_BACKGROUND_PROCESS'):
            try:
                with open(BACKGROUND_LOG_FILE, 'a') as f:
                    f.write(f"\n{error_msg}\n")
                    import traceback
                    f.write(traceback.format_exc())
                    f.write("\nScript exiting due to fatal error\n")
            except:
                pass
            
            # Clean up PID file if we're the background process
            try:
                if os.path.exists(BACKGROUND_PID_FILE):
                    os.remove(BACKGROUND_PID_FILE)
                    print("Cleaned up PID file due to error")
            except:
                pass
        
        # Re-raise the exception if not in background mode
        if not os.environ.get('IS_BACKGROUND_PROCESS'):
            raise
        else:
            import sys
            sys.exit(1)