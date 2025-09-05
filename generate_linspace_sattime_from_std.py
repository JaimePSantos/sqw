#!/usr/bin/env python3

"""
Generate Saturation Time Data from Standard Deviation Files - Linspace Version

This script calculates saturation times from existing standard deviation files
created by the linspace std generation script. It processes multiple deviation values
using configurable multiprocessing, calculating the saturation time for each
deviation where the std vs time curve reaches a plateau.

Simple Saturation Detection Method:
1. Calculate moving averages over different window sizes
2. Find where the relative change becomes small and stays small
3. Use multiple criteria to ensure robust detection
4. No complex derivatives or log-log analysis - just straightforward plateau detection

Key Features:
- Simple, robust saturation detection
- Configurable number of processes (each handling multiple deviation values)
- Smart file validation (checks if sattime files exist and have valid data)
- Automatic directory structure handling with N, steps, samples, and theta tracking
- Comprehensive logging for each process
- Graceful error handling and recovery

Usage:
    python generate_linspace_sattime_from_std.py

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

try:
    from scipy.signal import savgol_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ============================================================================
# ARCHIVING CONFIGURATION
# ============================================================================

CREATE_TAR = True  # If True, create per-chunk and main tar archives
ARCHIVE_DIR = "experiments_archive_linspace_sattime"

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

# Timeout configuration - Scale with problem size
BASE_TIMEOUT_PER_SAMPLE = 10  # seconds per sample for small N (sattime is faster than std)
TIMEOUT_SCALE_FACTOR = (N * steps) / 1000000  # Scale based on computational complexity
PROCESS_TIMEOUT = max(1800, int(BASE_TIMEOUT_PER_SAMPLE * samples * TIMEOUT_SCALE_FACTOR))  # Minimum 30 minutes

print(f"[TIMEOUT] Process timeout: {PROCESS_TIMEOUT} seconds ({PROCESS_TIMEOUT/3600:.1f} hours)")
print(f"[TIMEOUT] Based on N={N}, steps={steps}, samples={samples}")
print(f"[RESOURCE] Using {NUM_PROCESSES} processes out of {mp.cpu_count()} CPUs")

# ============================================================================
# SIMPLE SATURATION DETECTION PARAMETERS
# ============================================================================

# Simple saturation detection parameters
WINDOW_SIZES = [5, 10, 15]      # Different window sizes for moving averages
PLATEAU_THRESHOLD = 0.05        # Maximum relative change to consider "flat" (5%)
MIN_PLATEAU_LENGTH = 10         # Minimum number of consecutive flat points
TAIL_CHECK_FRACTION = 0.3       # Check last 30% of data for plateau

# Directory configuration
STD_BASE_DIR = "experiments_data_samples_linspace_std"
SATTIME_BASE_DIR = "experiments_data_samples_linspace_sattime"

# Create date-based logging directories
current_date = datetime.now().strftime("%d-%m-%y")
PROCESS_LOG_DIR = os.path.join("logs", current_date, "generate_sattime_linspace")

# Logging configuration
MASTER_LOG_FILE = os.path.join("logs", current_date, "generate_sattime_linspace", "sattime_generation_master.log")

# Global shutdown flag
SHUTDOWN_REQUESTED = False

# ============================================================================
# SATURATION TIME DETECTION PARAMETERS
# ============================================================================

# Smoothing method: 'savgol', 'moving_average', or None
SMOOTHING_METHOD = None  # Set to None to use raw data (no smoothing for real experimental data)

# Savitzky-Golay filter parameters (used if SMOOTHING_METHOD == 'savgol')
SAVGOL_WINDOW_FRAC = 0.11    # Window size as fraction of data length
SAVGOL_POLY_ORDER = 3        # Polynomial order for Savitzky-Golay filter

# Moving average parameters (used if SMOOTHING_METHOD == 'moving_average')
MOVING_AVG_WINDOW_FRAC = 0.05  # Window size as fraction of data length

# Threshold detection parameters
ADAPTIVE_THRESHOLD_FACTOR = 1.5  # Multiplier for MAD-based threshold (more sensitive)
FIXED_THRESHOLD = 0.02       # Fixed threshold fallback if adaptive fails (more sensitive)
RUN_LENGTH_FRAC = 0.03      # Minimum run length as fraction of data length (shorter run required)

# Noise estimation parameters
TAIL_FRACTION = 0.4        # Fraction of data from end to estimate noise (larger tail for better noise estimate)

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
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"process{process_id}_{dev_range_str}{theta_str}_sattime.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"process_{process_id}_sattime")
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

def validate_sattime_file(file_path):
    """
    Validate that a saturation time file exists and contains valid data.
    
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a valid dictionary with expected keys
        if isinstance(data, dict) and 'saturation_time' in data and 'saturation_value' in data:
            # Additional validation: check for reasonable saturation time values
            sat_time = data['saturation_time']
            sat_value = data['saturation_value']
            
            if sat_time is not None and sat_value is not None:
                if not np.isnan(sat_time) and not np.isnan(sat_value) and sat_time > 0 and sat_value > 0:
                    return True
        
        return False
        
    except (pickle.PickleError, EOFError, ValueError, TypeError) as e:
        return False

def load_std_data_for_dev(std_dir, logger):
    """
    Load standard deviation data for a single deviation from std files.
    
    Args:
        std_dir: Directory containing standard deviation files
        logger: Logger for this process
    
    Returns:
        np.ndarray: Array of standard deviation values vs time, or None if failed
    """
    try:
        logger.info(f"Loading standard deviation data from: {std_dir}")
        
        std_file = os.path.join(std_dir, "std_vs_time.pkl")
        
        if os.path.exists(std_file):
            try:
                with open(std_file, 'rb') as f:
                    std_data = pickle.load(f)
                
                # Convert to numpy array and validate
                std_array = np.array(std_data)
                if len(std_array) == 0:
                    logger.warning(f"Empty standard deviation data")
                    return None
                
                # Check for valid std values (should be non-negative)
                valid_mask = (std_array >= 0) & (~np.isnan(std_array)) & (~np.isinf(std_array))
                valid_count = np.sum(valid_mask)
                
                if valid_count == 0:
                    logger.warning(f"No valid standard deviation values found")
                    return None
                
                logger.info(f"Loaded {len(std_array)} std values, {valid_count} valid")
                return std_array
                
            except Exception as e:
                logger.warning(f"Failed to load standard deviation data: {e}")
                return None
        else:
            logger.warning(f"Standard deviation file not found: {std_file}")
            return None
        
    except Exception as e:
        logger.error(f"Error loading standard deviation data: {str(e)}")
        return None

# ============================================================================
# SIMPLE SATURATION TIME CALCULATION FUNCTIONS
# ============================================================================

def moving_average(data, window_size):
    """Calculate moving average of data."""
    if window_size >= len(data):
        return np.full(len(data), np.mean(data))
    
    result = np.zeros_like(data)
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        result[i] = np.mean(data[start:end])
    
    return result

def detect_saturation_simple(t, y, logger=None):
    """
    Saturation detection using slope analysis with linear regression.
    
    The method finds where the slope of std vs time becomes essentially zero,
    indicating that the standard deviation has stopped growing significantly.
    
    Args:
        t: Time array
        y: Standard deviation values array
        logger: Logger for debugging output
    
    Returns:
        tuple: (saturation_time, saturation_value, metadata_dict)
               Returns (np.nan, np.nan, {}) if no saturation found
    """
    try:
        # Input validation
        if len(t) != len(y) or len(t) < 20:
            if logger:
                logger.warning(f"Invalid input: t length={len(t)}, y length={len(y)}")
            return np.nan, np.nan, {}
        
        # Remove invalid values
        valid_mask = (y > 0) & (~np.isnan(y)) & (~np.isinf(y)) & (t >= 0) & (~np.isnan(t)) & (~np.isinf(t))
        if np.sum(valid_mask) < 15:
            if logger:
                logger.warning(f"Too few valid data points: {np.sum(valid_mask)}")
            return np.nan, np.nan, {}
        
        t_clean = t[valid_mask]
        y_clean = y[valid_mask]
        
        if logger:
            logger.debug(f"Analyzing {len(y_clean)} valid points for slope-based saturation")
        
        # Slope-based saturation detection parameters
        window_size = 15  # Window size for linear regression
        slope_threshold = 0.01  # Maximum absolute slope to consider as "flat"
        min_flat_length = 5  # Minimum number of consecutive flat windows
        
        if len(y_clean) < window_size + min_flat_length:
            if logger:
                logger.warning(f"Insufficient data for slope analysis: need {window_size + min_flat_length}, have {len(y_clean)}")
            return np.nan, np.nan, {}
        
        slopes = []
        slope_times = []
        
        # Calculate slopes using sliding window linear regression
        for i in range(len(y_clean) - window_size + 1):
            window_t = t_clean[i:i + window_size]
            window_y = y_clean[i:i + window_size]
            
            # Skip windows with insufficient variance
            if np.var(window_y) == 0:
                slopes.append(0.0)
                slope_times.append(t_clean[i + window_size // 2])
                continue
            
            # Linear regression: y = at + b, we want the slope 'a'
            try:
                slope, intercept = np.polyfit(window_t, window_y, 1)
                slopes.append(slope)
                slope_times.append(t_clean[i + window_size // 2])  # Time at center of window
                
                if logger and i % 50 == 0:  # Log every 50th calculation for debugging
                    logger.debug(f"Window {i}: t_center={slope_times[-1]:.1f}, slope={slope:.6f}")
                    
            except Exception as e:
                if logger:
                    logger.warning(f"Linear regression failed at window {i}: {e}")
                slopes.append(np.nan)
                slope_times.append(t_clean[i + window_size // 2])
        
        slopes = np.array(slopes)
        slope_times = np.array(slope_times)
        
        # Remove NaN slopes
        valid_slopes = ~np.isnan(slopes)
        slopes = slopes[valid_slopes]
        slope_times = slope_times[valid_slopes]
        
        if len(slopes) == 0:
            if logger:
                logger.warning("No valid slopes calculated")
            return np.nan, np.nan, {}
        
        # Find where absolute slope is below threshold
        flat_mask = np.abs(slopes) < slope_threshold
        
        if not np.any(flat_mask):
            if logger:
                min_abs_slope = np.min(np.abs(slopes))
                logger.warning(f"No flat region found. Min slope: {min_abs_slope:.6f}, threshold: {slope_threshold}")
            return np.nan, np.nan, {'method': 'slope_regression', 'error': 'no_flat_region_found', 
                                    'min_slope': np.min(np.abs(slopes)), 'threshold': slope_threshold}
        
        # Find first sustained flat region
        flat_indices = np.where(flat_mask)[0]
        
        # Look for consecutive flat windows
        consecutive_count = 1
        start_idx = flat_indices[0]
        
        for i in range(1, len(flat_indices)):
            if flat_indices[i] == flat_indices[i-1] + 1:
                consecutive_count += 1
            else:
                if consecutive_count >= min_flat_length:
                    break
                consecutive_count = 1
                start_idx = flat_indices[i]
        
        if consecutive_count < min_flat_length:
            if logger:
                logger.warning(f"Insufficient consecutive flat length: {consecutive_count} < {min_flat_length}")
            return np.nan, np.nan, {'method': 'slope_regression', 'error': 'insufficient_flat_length',
                                    'max_consecutive': consecutive_count, 'required': min_flat_length}
        
        # Saturation occurs at the start of the first sustained flat region
        saturation_time = slope_times[start_idx]
        
        # Find the corresponding y value
        sat_time_idx = np.argmin(np.abs(t_clean - saturation_time))
        saturation_value = y_clean[sat_time_idx]
        
        if logger:
            logger.info(f"Slope-based saturation detected: t={saturation_time:.2f}, value={saturation_value:.4f}, slope={slopes[start_idx]:.6f}")
        
        metadata = {
            'method': 'slope_regression',
            'window_size': window_size,
            'slope_threshold': slope_threshold,
            'min_flat_length': min_flat_length,
            'saturation_slope': slopes[start_idx],
            'consecutive_flat_windows': consecutive_count,
            'total_slopes_calculated': len(slopes),
            'flat_regions_found': np.sum(flat_mask)
        }
        
        return saturation_time, saturation_value, metadata
        
    except Exception as e:
        if logger:
            logger.error(f"Error in slope-based saturation detection: {e}")
        return np.nan, np.nan, {'method': 'slope_regression', 'error': str(e)}

def generate_sattime_for_dev_chunk(chunk_args):
    """
    Worker function to generate saturation time data for a chunk of deviation values.
    
    Args:
        chunk_args: Tuple containing (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, std_base_dir, sattime_base_dir)
    
    Returns:
        dict: Results from the sattime generation process
    """
    dev_chunk, process_id, N, steps, samples_count, theta_param, shutdown_flag, std_base_dir, sattime_base_dir = chunk_args
    
    # Import required modules (each process needs its own imports)
    import gc  # For garbage collection
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(process_id, dev_chunk, theta_param)
    
    try:
        logger.info(f"=== SATURATION TIME GENERATION STARTED ===")
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
            
            # Create directory structure for std and sattime
            theta_folder = format_theta_for_directory(theta_param)
            
            # Std source directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            std_exp_dir = os.path.join(std_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            # Sattime target directory: base_dir/static_noise_linspace/N_xxxx/theta_folder/dev_folder
            sattime_exp_dir = os.path.join(sattime_base_dir, "static_noise_linspace", f"N_{N}", theta_folder, dev_folder)
            
            logger.info(f"Std source: {std_exp_dir}")
            logger.info(f"Sattime target: {sattime_exp_dir}")
            
            # Check if std directory exists
            if not os.path.exists(std_exp_dir):
                logger.error(f"Std directory not found: {std_exp_dir}")
                continue
            
            # Check if sattime file already exists and is valid
            os.makedirs(sattime_exp_dir, exist_ok=True)
            sattime_file = os.path.join(sattime_exp_dir, "saturation_time.pkl")
            
            skipped = False
            if validate_sattime_file(sattime_file):
                logger.info(f"Valid sattime file already exists: {sattime_file}")
                skipped = True
                chunk_skipped_count += 1
            else:
                # Load standard deviation data
                std_data = load_std_data_for_dev(std_exp_dir, logger)
                if std_data is None or len(std_data) == 0:
                    logger.error(f"No standard deviation data loaded")
                    continue
                
                # Create time array
                t = np.arange(len(std_data))
                
                # Calculate saturation time using simple method
                logger.info(f"Calculating saturation time for {len(std_data)} time steps...")
                sat_time, sat_value, metadata = detect_saturation_simple(t, std_data, logger=logger)
                
                # Prepare result data
                result_data = {
                    'saturation_time': sat_time,
                    'saturation_value': sat_value,
                    'metadata': metadata,
                    'std_data': std_data,
                    'time_array': t,
                    'parameters': {
                        'N': N,
                        'steps': steps,
                        'samples': samples_count,
                        'theta': theta_param,
                        'deviation': dev
                    }
                }
                
                # Save saturation time data
                with open(sattime_file, 'wb') as f:
                    pickle.dump(result_data, f)
                logger.info(f"Saturation time data saved to: {sattime_file}")
                
                if not np.isnan(sat_time):
                    logger.info(f"Saturation time: {sat_time:.2f}, Saturation value: {sat_value:.6f}")
                    logger.info(f"Metadata: {metadata}")
                else:
                    logger.warning(f"No saturation detected for this deviation")
                
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
        
        logger.info(f"=== SATTIME GENERATION COMPLETED ===")
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
        error_msg = f"Error in sattime generation for process {process_id}: {str(e)}"
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
    """Main execution function for saturation time generation."""
    
    print("=== SATURATION TIME GENERATION - LINSPACE VERSION ===")
    print(f"System parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    print(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    print(f"Total deviations: {len(devs)}")
    print(f"Multiprocessing: {NUM_PROCESSES} processes")
    print(f"Std source: {STD_BASE_DIR}")
    print(f"Sattime target: {SATTIME_BASE_DIR}")
    print(f"SciPy available: {HAS_SCIPY}")
    print("=" * 70)
    
    # Setup master logging
    master_logger = setup_master_logging()
    master_logger.info("=== SATURATION TIME GENERATION - LINSPACE VERSION STARTED ===")
    master_logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.6f}")
    master_logger.info(f"Deviation range: {DEV_MIN} to {DEV_MAX} with {DEV_COUNT} values")
    master_logger.info(f"Total deviations: {len(devs)}")
    master_logger.info(f"Processes: {NUM_PROCESSES}")
    master_logger.info(f"SciPy available: {HAS_SCIPY}")
    
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
            args = (dev_chunk, process_id, N, steps, samples, theta, shutdown_flag, STD_BASE_DIR, SATTIME_BASE_DIR)
            process_args.append(args)
    
    print(f"\nStarting {len(process_args)} processes for sattime generation...")
    
    process_results = []
    
    try:
        with ProcessPoolExecutor(max_workers=len(process_args)) as executor:
            # Submit all processes
            future_to_process = {}
            for args in process_args:
                future = executor.submit(generate_sattime_for_dev_chunk, args)
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

        main_tar_name = f"experiments_data_sattime_linspace_N{N}_samples{samples}_theta{theta:.6f}_{timestamp}.tar"
        main_tar_path = os.path.join(ARCHIVE_DIR, main_tar_name)
        
        print(f"Creating archive: {main_tar_path}")
        master_logger.info(f"Creating archive: {main_tar_path}")
        
        try:
            with tarfile.open(main_tar_path, "w") as tar:
                tar.add(SATTIME_BASE_DIR, arcname=os.path.basename(SATTIME_BASE_DIR))
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
