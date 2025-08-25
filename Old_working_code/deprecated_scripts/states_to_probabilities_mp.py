#!/usr/bin/env python3

"""
States to Probabilities Multiprocessor Converter

This script loads quantum walk state samples and converts them to probability distributions
using multiprocessing for efficient parallel computation. After conversion, it creates
archives of the results.

Features:
- Multiprocessing for parallel probability calculation across deviations
- Memory-efficient streaming approach for large datasets
- Automatic sample detection and loading
- Mean probability distribution calculation from multiple samples
- Tar archive creation with multiprocessing support
- Comprehensive logging and progress tracking
- Resource monitoring and error recovery
- Support for static noise experiment structure

Execution Flow:
1. Load quantum state samples from experiments_data_samples
2. Convert states to probability distributions using amp2prob
3. Calculate mean probability distributions across samples
4. Save results to experiments_data_samples_probDist
5. Create compressed archives of results

Author: Extracted from static_cluster_logged_mp.py
"""

import os
import pickle
import time
import numpy as np
import math
import logging
import signal
import multiprocessing as mp
import tarfile
import gc
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Multiprocessing configuration
MAX_PROCESSES = min(mp.cpu_count(), 8)  # Conservative approach
PROCESS_TIMEOUT = 7200  # 2 hours per process
MEAN_PROB_TIMEOUT = 10800  # 3 hours for mean probability calculation

# Archive configuration
CREATE_TAR_ARCHIVE = True
USE_MULTIPROCESS_ARCHIVING = True
MAX_ARCHIVE_PROCESSES = min(5, mp.cpu_count())
EXCLUDE_SAMPLES_FROM_ARCHIVE = True  # Only archive processed data

# Logging configuration
PROCESS_LOG_DIR = "process_logs_prob_conversion"
MASTER_LOG_FILE = "states_to_probabilities_conversion.log"

# Memory management
PROGRESS_UPDATE_INTERVAL = 100  # Log progress every N steps
RESOURCE_CHECK_INTERVAL = 300  # Check resources every 5 minutes

# Global shutdown flag
SHUTDOWN_REQUESTED = False

# ============================================================================
# SIGNAL HANDLING
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
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal (Unix)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_logging():
    """Setup master logging"""
    # Create logger
    logger = logging.getLogger("states_to_prob_master")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(MASTER_LOG_FILE, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] [MASTER] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def setup_process_logging(dev_value, process_id):
    """Setup logging for individual processes"""
    os.makedirs(PROCESS_LOG_DIR, exist_ok=True)
    
    # Format dev_value for filename
    if isinstance(dev_value, (tuple, list)) and len(dev_value) == 2:
        min_val, max_val = dev_value
        dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        dev_str = f"{float(dev_value):.3f}"
    
    log_filename = os.path.join(PROCESS_LOG_DIR, f"prob_conversion_dev_{dev_str}_pid_{process_id}.log")
    
    # Create logger for this process
    logger = logging.getLogger(f"prob_conv_dev_{dev_str}")
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
            warning_msg = f"{prefix} WARNING: High memory usage ({memory.percent:.1f}%)"
            if logger:
                logger.warning(warning_msg)
            else:
                print(warning_msg)
        
        if cpu_percent > 95:
            warning_msg = f"{prefix} WARNING: High CPU usage ({cpu_percent:.1f}%)"
            if logger:
                logger.warning(warning_msg)
            else:
                print(warning_msg)
                
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

# ============================================================================
# DUMMY TESSELLATION FUNCTION (for compatibility)
# ============================================================================

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

# ============================================================================
# SAMPLE DETECTION AND LOADING
# ============================================================================

def detect_experiment_structure(base_dir="experiments_data_samples"):
    """
    Automatically detect experiment structure and available samples.
    
    Returns:
        dict: Experiment metadata including N, steps, deviations, samples
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Walk through directory structure to find experiments
    experiments = {}
    
    for root, dirs, files in os.walk(base_dir):
        # Look for N_ directories
        for dir_name in dirs:
            if dir_name.startswith("N_"):
                try:
                    N = int(dir_name[2:])  # Extract N value
                    
                    # Get experiment path components
                    rel_path = os.path.relpath(root, base_dir)
                    path_parts = rel_path.split(os.sep) if rel_path != "." else []
                    
                    # Extract deviation and theta information from path
                    dev_info = None
                    theta_info = None
                    
                    for part in path_parts:
                        if part.startswith("dev_"):
                            dev_info = part
                        elif part.startswith("theta_"):
                            theta_info = part
                    
                    # Count samples in this N directory
                    n_dir_path = os.path.join(root, dir_name)
                    sample_count = count_samples_in_directory(n_dir_path)
                    
                    if sample_count > 0:
                        exp_key = (N, dev_info, theta_info)
                        experiments[exp_key] = {
                            "N": N,
                            "path": n_dir_path,
                            "dev_info": dev_info,
                            "theta_info": theta_info,
                            "sample_count": sample_count,
                            "deviation": parse_deviation_from_path(dev_info) if dev_info else (0, 0)
                        }
                        
                except ValueError:
                    continue  # Skip invalid N directories
    
    if not experiments:
        raise ValueError(f"No valid experiments found in {base_dir}")
    
    # Group by N and extract metadata
    n_values = list(set(exp["N"] for exp in experiments.values()))
    if len(n_values) > 1:
        print(f"[WARNING] Multiple N values found: {n_values}. Using largest: {max(n_values)}")
        N = max(n_values)
    else:
        N = n_values[0]
    
    # Extract experiments for the selected N
    n_experiments = {k: v for k, v in experiments.items() if v["N"] == N}
    
    # Extract unique deviations and theta
    deviations = list(set(exp["deviation"] for exp in n_experiments.values()))
    theta_values = list(set(exp["theta_info"] for exp in n_experiments.values() if exp["theta_info"]))
    
    # Parse theta value
    theta = None
    if theta_values:
        theta_str = theta_values[0]  # Assume single theta
        if theta_str.startswith("theta_"):
            try:
                theta = float(theta_str[6:])
            except ValueError:
                theta = math.pi/3  # Default
    else:
        theta = math.pi/3  # Default
    
    # Estimate steps from sample files
    steps = estimate_steps_from_samples(list(n_experiments.values())[0]["path"])
    
    # Get sample counts
    samples = max(exp["sample_count"] for exp in n_experiments.values())
    
    return {
        "N": N,
        "steps": steps,
        "theta": theta,
        "deviations": sorted(deviations),
        "samples": samples,
        "experiments": n_experiments,
        "base_dir": base_dir
    }

def parse_deviation_from_path(dev_info):
    """Parse deviation values from directory name"""
    if not dev_info or not dev_info.startswith("dev_"):
        return (0, 0)
    
    # Handle different formats: dev_min0.000_max0.200, dev_0.200, etc.
    dev_part = dev_info[4:]  # Remove "dev_" prefix
    
    if "min" in dev_part and "max" in dev_part:
        # Format: min0.000_max0.200
        parts = dev_part.split("_")
        min_val = max_val = 0
        for part in parts:
            if part.startswith("min"):
                min_val = float(part[3:])
            elif part.startswith("max"):
                max_val = float(part[3:])
        return (min_val, max_val)
    else:
        # Simple format: just a number
        try:
            val = float(dev_part)
            return (0, val)  # Assume range [0, val]
        except ValueError:
            return (0, 0)

def count_samples_in_directory(n_dir_path):
    """Count the number of samples in an N directory"""
    if not os.path.exists(n_dir_path):
        return 0
    
    # Look for step directories
    step_dirs = [d for d in os.listdir(n_dir_path) if d.startswith("step_") and os.path.isdir(os.path.join(n_dir_path, d))]
    
    if not step_dirs:
        return 0
    
    # Check first step directory for sample files
    first_step_dir = os.path.join(n_dir_path, step_dirs[0])
    sample_files = [f for f in os.listdir(first_step_dir) if f.startswith("final_step_") and f.endswith(".pkl")]
    
    # Extract sample indices
    sample_indices = set()
    for filename in sample_files:
        # Format: final_step_{step_idx}_sample{sample_idx}.pkl
        if "_sample" in filename:
            try:
                sample_part = filename.split("_sample")[1].split(".")[0]
                sample_idx = int(sample_part)
                sample_indices.add(sample_idx)
            except (IndexError, ValueError):
                continue
    
    return len(sample_indices)

def estimate_steps_from_samples(n_dir_path):
    """Estimate number of steps from sample files"""
    if not os.path.exists(n_dir_path):
        return 0
    
    # Count step directories
    step_dirs = [d for d in os.listdir(n_dir_path) if d.startswith("step_") and os.path.isdir(os.path.join(n_dir_path, d))]
    
    if not step_dirs:
        return 0
    
    # Extract step indices
    step_indices = []
    for step_dir in step_dirs:
        try:
            step_idx = int(step_dir[5:])  # Remove "step_" prefix
            step_indices.append(step_idx)
        except ValueError:
            continue
    
    return max(step_indices) + 1 if step_indices else 0

# ============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================================

def convert_states_to_probabilities_worker(worker_args):
    """
    Worker function to convert quantum states to probability distributions for a single deviation.
    
    Args:
        worker_args: Tuple containing (dev, process_id, N, steps, samples, source_base_dir, target_base_dir, theta, exp_path)
    
    Returns:
        dict: Results including success status, processing statistics, and log file path
    """
    dev, process_id, N, steps, samples, source_base_dir, target_base_dir, theta, exp_path = worker_args
    
    # Setup logging for this process
    logger, log_file = setup_process_logging(dev, process_id)
    
    try:
        logger.info(f"Starting probability conversion for deviation {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}")
        logger.info(f"Source path: {exp_path}")
        
        # Import required modules (each process needs its own imports)
        from sqw.states import amp2prob
        from smart_loading_static import find_experiment_dir_flexible
        
        dev_start_time = time.time()
        
        # Handle deviation format for target directory
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            min_val, max_val = dev
            has_noise = max_val > 0
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            has_noise = dev > 0
            dev_str = f"{dev:.3f}"
        
        # Get target directory using the same structure as source
        noise_params = [dev]
        target_exp_dir, _ = find_experiment_dir_flexible(
            dummy_tesselation_func, has_noise, N, 
            noise_params=noise_params, noise_type="static_noise", 
            base_dir=target_base_dir, theta=theta
        )
        
        os.makedirs(target_exp_dir, exist_ok=True)
        logger.info(f"Target directory: {target_exp_dir}")
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        
        processed_steps = 0
        skipped_steps = 0
        failed_steps = 0
        last_resource_check = time.time()
        
        for step_idx in range(steps):
            # Check for shutdown signal
            if SHUTDOWN_REQUESTED:
                logger.warning("Shutdown requested, stopping conversion")
                break
            
            # Check if mean probability file already exists
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(target_exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                skipped_steps += 1
                if step_idx % PROGRESS_UPDATE_INTERVAL == 0:
                    logger.info(f"    Step {step_idx+1}/{steps} already exists, skipping")
                continue
            
            # Progress logging
            if step_idx % PROGRESS_UPDATE_INTERVAL == 0:
                logger.info(f"    Processing step {step_idx+1}/{steps}... (processed: {processed_steps}, skipped: {skipped_steps})")
            
            # Resource monitoring
            current_time = time.time()
            if current_time - last_resource_check >= RESOURCE_CHECK_INTERVAL:
                log_system_resources(logger, "[WORKER]")
                last_resource_check = current_time
            
            # Load samples for this step
            step_dir = os.path.join(exp_path, f"step_{step_idx}")
            
            if not os.path.exists(step_dir):
                logger.warning(f"Step directory not found: {step_dir}")
                failed_steps += 1
                continue
            
            # Load all available samples for this step
            sample_states = []
            loaded_samples = 0
            
            for sample_idx in range(samples):
                filename = f"final_step_{step_idx}_sample{sample_idx}.pkl"
                filepath = os.path.join(step_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            state = pickle.load(f)
                        sample_states.append(state)
                        loaded_samples += 1
                    except Exception as e:
                        logger.warning(f"Failed to load sample {sample_idx} for step {step_idx}: {e}")
                else:
                    logger.debug(f"Sample file not found: {filepath}")
            
            if not sample_states:
                logger.warning(f"No valid samples found for step {step_idx}")
                failed_steps += 1
                continue
            
            try:
                # Convert quantum states to probability distributions
                prob_distributions = []
                for i, state in enumerate(sample_states):
                    try:
                        prob_dist = amp2prob(state)  # |amplitude|^2
                        prob_distributions.append(prob_dist)
                    except Exception as e:
                        logger.warning(f"Failed to convert state {i} to probability: {e}")
                    finally:
                        # Clear state from memory immediately
                        sample_states[i] = None
                
                if not prob_distributions:
                    logger.warning(f"No valid probability distributions for step {step_idx}")
                    failed_steps += 1
                    continue
                
                # Calculate mean probability distribution
                mean_prob_dist = np.mean(prob_distributions, axis=0)
                
                # Clear prob_distributions to save memory
                del prob_distributions
                gc.collect()  # Force garbage collection
                
                # Save mean probability distribution
                with open(mean_filepath, "wb") as f:
                    pickle.dump(mean_prob_dist, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                processed_steps += 1
                
                # Detailed logging for significant milestones
                if step_idx % PROGRESS_UPDATE_INTERVAL == 0 or step_idx == steps - 1:
                    logger.info(f"    Step {step_idx+1}/{steps} completed (samples: {loaded_samples}/{samples})")
                    
            except Exception as e:
                logger.error(f"Failed to process step {step_idx}: {e}")
                failed_steps += 1
                continue
        
        dev_time = time.time() - dev_start_time
        
        # Log completion summary
        logger.info(f"Deviation {dev_str} conversion completed:")
        logger.info(f"  Processing time: {dev_time:.1f}s")
        logger.info(f"  Processed steps: {processed_steps}/{steps}")
        logger.info(f"  Skipped steps: {skipped_steps}")
        logger.info(f"  Failed steps: {failed_steps}")
        logger.info(f"  Success rate: {(processed_steps/(steps-skipped_steps)*100):.1f}%" if steps > skipped_steps else "N/A")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": True,
            "processed_steps": processed_steps,
            "skipped_steps": skipped_steps,
            "failed_steps": failed_steps,
            "total_steps": steps,
            "processing_time": dev_time,
            "log_file": log_file,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Critical error in probability conversion for dev {dev}: {str(e)}"
        logger.error(error_msg)
        logger.error("Exception details:", exc_info=True)
        
        return {
            "dev": dev,
            "process_id": process_id,
            "success": False,
            "processed_steps": 0,
            "skipped_steps": 0,
            "failed_steps": 0,
            "total_steps": steps,
            "processing_time": 0,
            "log_file": log_file,
            "error": error_msg
        }

# ============================================================================
# ARCHIVING FUNCTIONS
# ============================================================================

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

def create_probability_archive(N, samples, use_multiprocess=True, max_archive_processes=5, logger=None):
    """
    Create a tar archive of probability distribution data.
    
    Args:
        N: System size
        samples: Number of samples per deviation
        use_multiprocess: Whether to use multiprocess archiving
        max_archive_processes: Maximum number of processes for archiving
        logger: Optional logger for logging archive operations
    
    Returns:
        str or None: Archive filename if successful, None if failed
    """
    def log_and_print(message, level="info"):
        """Helper function to log messages"""
        if logger:
            clean_msg = message.replace("[ARCHIVE] ", "").replace("[WARNING] ", "").replace("[OK] ", "").replace("[ERROR] ", "")
            if level == "info":
                logger.info(clean_msg)
            elif level == "warning":
                logger.warning(clean_msg)
            elif level == "error":
                logger.error(clean_msg)
        else:
            print(message)
    
    try:
        log_and_print("[ARCHIVE] Creating tar archive of probability distribution data...")
        
        # Create archive filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_archive_name = f"probability_distributions_N{N}_samples{samples}_{timestamp}.tar.gz"
        
        # Define data folders to archive (only processed data)
        data_folders = ["experiments_data_samples_probDist"]
        
        n_folder_name = f"N_{N}"
        all_folders_to_archive = []
        
        # Find all folders containing N_{N} folders
        for data_folder in data_folders:
            if not os.path.exists(data_folder):
                log_and_print(f"[INFO] Data folder '{data_folder}' not found - skipping")
                continue
                
            log_and_print(f"[ARCHIVE] Scanning folder: {os.path.abspath(data_folder)}")
            
            folder_found = False
            for root, dirs, files in os.walk(data_folder):
                if n_folder_name in dirs:
                    relative_root = os.path.relpath(root, data_folder)
                    if relative_root == ".":
                        folder_path = n_folder_name
                    else:
                        folder_path = os.path.join(relative_root, n_folder_name)
                    
                    full_path = os.path.join(data_folder, folder_path)
                    archive_path = os.path.join(data_folder, folder_path)
                    all_folders_to_archive.append((full_path, archive_path))
                    log_and_print(f"  Found: {archive_path}")
                    folder_found = True
            
            if not folder_found:
                log_and_print(f"[INFO] No '{n_folder_name}' folders found in {data_folder}")
        
        if not all_folders_to_archive:
            log_and_print(f"[WARNING] No folders found containing '{n_folder_name}' - skipping archive creation", "warning")
            return None
        
        total_folders = len(all_folders_to_archive)
        log_and_print(f"[ARCHIVE] Found {total_folders} folders to archive")
        
        # Determine archiving method
        if not use_multiprocess or total_folders <= 1:
            log_and_print(f"[ARCHIVE] Using single-process archiving")
            
            # Create the tar archive with all folders
            with tarfile.open(final_archive_name, "w:gz") as tar:
                for i, (full_path, archive_path) in enumerate(all_folders_to_archive, 1):
                    log_and_print(f"  [{i}/{total_folders}] Adding: {archive_path}")
                    tar.add(full_path, arcname=archive_path)
        
        else:
            # Multiprocess archiving
            if max_archive_processes is None:
                max_archive_processes = min(total_folders, mp.cpu_count())
            
            log_and_print(f"[ARCHIVE] Using multiprocess archiving with {max_archive_processes} processes")
            
            # Prepare arguments for multiprocessing
            temp_archives = []
            archive_args = []
            
            for i, (full_path, archive_path) in enumerate(all_folders_to_archive):
                temp_archive_name = f"temp_prob_archive_N{N}_{i}_{timestamp}.tar.gz"
                temp_archives.append(temp_archive_name)
                archive_args.append((full_path, archive_path, temp_archive_name))
            
            # Create individual archives in parallel
            successful_archives = []
            failed_archives = []
            
            try:
                with ProcessPoolExecutor(max_workers=max_archive_processes) as executor:
                    future_to_args = {}
                    for args in archive_args:
                        future = executor.submit(create_single_archive, args)
                        future_to_args[future] = args
                    
                    completed = 0
                    for future in as_completed(future_to_args):
                        completed += 1
                        args = future_to_args[future]
                        try:
                            result = future.result()
                            if result["success"]:
                                successful_archives.append(result)
                                log_and_print(f"  [{completed}/{total_folders}] Created {result['archive_name']} ({result['size_mb']:.1f} MB)")
                            else:
                                failed_archives.append(result)
                                log_and_print(f"  [{completed}/{total_folders}] Failed {result['archive_name']}: {result['error']}", "error")
                        except Exception as e:
                            failed_archives.append({
                                "success": False,
                                "archive_name": args[2],
                                "archive_path": args[1],
                                "size_mb": 0,
                                "error": str(e)
                            })
                            log_and_print(f"  [{completed}/{total_folders}] Exception: {e}", "error")
            
            except Exception as e:
                log_and_print(f"[ERROR] Critical error in multiprocess archiving: {e}", "error")
                # Clean up temp files
                for temp_name in temp_archives:
                    try:
                        if os.path.exists(temp_name):
                            os.remove(temp_name)
                    except:
                        pass
                raise
            
            if len(successful_archives) == 0:
                log_and_print("[ERROR] No archives were created successfully", "error")
                return None
            
            # Combine successful archives
            log_and_print(f"[ARCHIVE] Combining {len(successful_archives)} archives into final archive")
            
            try:
                with tarfile.open(final_archive_name, "w:gz") as final_tar:
                    for i, result in enumerate(successful_archives, 1):
                        temp_archive_name = result["archive_name"]
                        log_and_print(f"  [{i}/{len(successful_archives)}] Merging {temp_archive_name}")
                        
                        with tarfile.open(temp_archive_name, "r:gz") as temp_tar:
                            for member in temp_tar.getmembers():
                                fileobj = temp_tar.extractfile(member)
                                if fileobj:
                                    final_tar.addfile(member, fileobj)
                                else:
                                    final_tar.addfile(member)
                
                # Clean up temporary archives
                for result in successful_archives:
                    try:
                        os.remove(result["archive_name"])
                    except Exception as e:
                        log_and_print(f"Warning: Could not remove {result['archive_name']}: {e}", "warning")
                        
            except Exception as e:
                log_and_print(f"[ERROR] Failed to combine archives: {e}", "error")
                return None
        
        # Get final archive size and report success
        if os.path.exists(final_archive_name):
            archive_size = os.path.getsize(final_archive_name)
            size_mb = archive_size / (1024 * 1024)
            log_and_print(f"[OK] Archive created successfully: {final_archive_name} ({size_mb:.1f} MB)")
            return final_archive_name
        else:
            log_and_print("[ERROR] Archive file was not created", "error")
            return None
        
    except Exception as e:
        log_and_print(f"[ERROR] Failed to create archive: {e}", "error")
        return None

# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_states_to_probabilities_conversion():
    """
    Main function to convert quantum walk states to probability distributions.
    
    Returns:
        dict: Execution results and statistics
    """
    print("=" * 80)
    print("QUANTUM WALK STATES TO PROBABILITIES CONVERTER")
    print("=" * 80)
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("STATES TO PROBABILITIES CONVERSION STARTED")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Auto-detect experiment structure
        print("\n[DETECTION] Auto-detecting experiment structure...")
        logger.info("Auto-detecting experiment structure...")
        
        exp_metadata = detect_experiment_structure()
        
        N = exp_metadata["N"]
        steps = exp_metadata["steps"]
        theta = exp_metadata["theta"]
        deviations = exp_metadata["deviations"]
        samples = exp_metadata["samples"]
        experiments = exp_metadata["experiments"]
        source_base_dir = exp_metadata["base_dir"]
        
        print(f"[DETECTION] Found experiment with:")
        print(f"  System size (N): {N}")
        print(f"  Time steps: {steps}")
        print(f"  Theta: {theta:.6f}")
        print(f"  Deviations: {len(deviations)} values")
        print(f"  Samples per deviation: {samples}")
        print(f"  Source directory: {source_base_dir}")
        
        logger.info(f"Experiment parameters: N={N}, steps={steps}, theta={theta:.6f}")
        logger.info(f"Deviations: {deviations}")
        logger.info(f"Samples: {samples}")
        
        # Configuration
        target_base_dir = "experiments_data_samples_probDist"
        max_processes = min(len(deviations), MAX_PROCESSES)
        
        print(f"\n[CONFIG] Conversion configuration:")
        print(f"  Target directory: {target_base_dir}")
        print(f"  Max processes: {max_processes}")
        print(f"  Process timeout: {PROCESS_TIMEOUT} seconds")
        print(f"  Archive creation: {CREATE_TAR_ARCHIVE}")
        if CREATE_TAR_ARCHIVE:
            print(f"  Multiprocess archiving: {USE_MULTIPROCESS_ARCHIVING}")
            print(f"  Max archive processes: {MAX_ARCHIVE_PROCESSES}")
        
        logger.info(f"Configuration: max_processes={max_processes}, timeout={PROCESS_TIMEOUT}s")
        
        # Log system resources
        log_system_resources(logger, "[SYSTEM]")
        
        # Prepare worker arguments
        print(f"\n[MULTIPROCESS] Preparing {len(deviations)} worker processes...")
        logger.info("=" * 40)
        logger.info("MULTIPROCESS PROBABILITY CONVERSION")
        logger.info("=" * 40)
        
        worker_args = []
        for process_id, (exp_key, exp_info) in enumerate(experiments.items()):
            dev = exp_info["deviation"]
            exp_path = exp_info["path"]
            
            args = (dev, process_id, N, steps, samples, source_base_dir, target_base_dir, theta, exp_path)
            worker_args.append(args)
        
        # Execute multiprocessing conversion
        process_results = []
        successful_processes = 0
        failed_processes = 0
        
        print(f"[MULTIPROCESS] Launching {len(worker_args)} processes...")
        logger.info(f"Launching {len(worker_args)} processes...")
        
        try:
            with ProcessPoolExecutor(max_workers=max_processes) as executor:
                # Submit all jobs
                future_to_args = {}
                for args in worker_args:
                    future = executor.submit(convert_states_to_probabilities_worker, args)
                    future_to_args[future] = args
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_args, timeout=PROCESS_TIMEOUT):
                    completed += 1
                    args = future_to_args[future]
                    dev = args[0]
                    
                    try:
                        result = future.result()
                        process_results.append(result)
                        
                        if result["success"]:
                            successful_processes += 1
                            print(f"  [{completed}/{len(worker_args)}] ✓ Dev {dev}: {result['processed_steps']}/{result['total_steps']} steps in {result['processing_time']:.1f}s")
                            logger.info(f"Process {result['process_id']} completed successfully: {result['processed_steps']}/{result['total_steps']} steps")
                        else:
                            failed_processes += 1
                            print(f"  [{completed}/{len(worker_args)}] ✗ Dev {dev}: FAILED - {result['error']}")
                            logger.error(f"Process {result['process_id']} failed: {result['error']}")
                            
                    except Exception as e:
                        failed_processes += 1
                        error_result = {
                            "dev": dev,
                            "process_id": args[1],
                            "success": False,
                            "processed_steps": 0,
                            "skipped_steps": 0,
                            "failed_steps": 0,
                            "total_steps": steps,
                            "processing_time": 0,
                            "log_file": "",
                            "error": str(e)
                        }
                        process_results.append(error_result)
                        print(f"  [{completed}/{len(worker_args)}] ✗ Dev {dev}: EXCEPTION - {e}")
                        logger.error(f"Process exception for dev {dev}: {e}")
        
        except TimeoutError:
            logger.error("Process execution timed out")
            print(f"[ERROR] Process execution timed out after {PROCESS_TIMEOUT} seconds")
        except Exception as e:
            logger.error(f"Critical error in multiprocessing: {e}")
            print(f"[ERROR] Critical error in multiprocessing: {e}")
            raise
        
        conversion_time = time.time() - start_time
        
        # Log results summary
        print(f"\n[RESULTS] Conversion completed in {conversion_time:.1f}s")
        print(f"  Successful processes: {successful_processes}/{len(deviations)}")
        print(f"  Failed processes: {failed_processes}")
        
        logger.info("=" * 40)
        logger.info("CONVERSION COMPLETED")
        logger.info("=" * 40)
        logger.info(f"Total time: {conversion_time:.1f}s")
        logger.info(f"Results: {successful_processes} successful, {failed_processes} failed processes")
        
        # Process-level statistics
        total_processed_steps = sum(r["processed_steps"] for r in process_results if r["success"])
        total_skipped_steps = sum(r["skipped_steps"] for r in process_results if r["success"])
        total_failed_steps = sum(r["failed_steps"] for r in process_results if r["success"])
        
        print(f"\n[STATISTICS] Overall statistics:")
        print(f"  Total processed steps: {total_processed_steps}")
        print(f"  Total skipped steps: {total_skipped_steps}")
        print(f"  Total failed steps: {total_failed_steps}")
        
        # Create archive if enabled and successful
        archive_name = None
        if CREATE_TAR_ARCHIVE and successful_processes > 0:
            print(f"\n[ARCHIVE] Creating archive...")
            logger.info("Creating probability distribution archive...")
            
            archive_name = create_probability_archive(
                N, samples, USE_MULTIPROCESS_ARCHIVING, MAX_ARCHIVE_PROCESSES, logger
            )
            
            if archive_name:
                print(f"[ARCHIVE] ✓ Archive created: {archive_name}")
                logger.info(f"Archive created successfully: {archive_name}")
            else:
                print(f"[ARCHIVE] ✗ Archive creation failed")
                logger.error("Archive creation failed")
        
        total_time = time.time() - start_time
        
        print(f"\n[COMPLETE] ✓ States to probabilities conversion completed!")
        print(f"  Total execution time: {total_time:.1f}s")
        print(f"  Successful conversions: {successful_processes}/{len(deviations)}")
        print(f"  Log files: {MASTER_LOG_FILE}, {PROCESS_LOG_DIR}/")
        if archive_name:
            print(f"  Archive: {archive_name}")
        
        logger.info(f"FINAL RESULTS: Total time={total_time:.1f}s, Success rate={successful_processes}/{len(deviations)}")
        
        return {
            "success": True,
            "total_time": total_time,
            "conversion_time": conversion_time,
            "N": N,
            "steps": steps,
            "theta": theta,
            "deviations": deviations,
            "samples": samples,
            "successful_processes": successful_processes,
            "failed_processes": failed_processes,
            "total_processed_steps": total_processed_steps,
            "total_skipped_steps": total_skipped_steps,
            "total_failed_steps": total_failed_steps,
            "process_results": process_results,
            "archive_name": archive_name,
            "master_log_file": MASTER_LOG_FILE,
            "process_log_dir": PROCESS_LOG_DIR
        }
        
    except Exception as e:
        error_msg = f"Critical error in states to probabilities conversion: {e}"
        print(f"\n[ERROR] {error_msg}")
        logger.error(error_msg)
        logger.error("Exception details:", exc_info=True)
        
        total_time = time.time() - start_time
        logger.info(f"Conversion failed after {total_time:.1f}s")
        
        return {
            "success": False,
            "total_time": total_time,
            "error": error_msg,
            "master_log_file": MASTER_LOG_FILE,
            "process_log_dir": PROCESS_LOG_DIR
        }

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    try:
        result = run_states_to_probabilities_conversion()
        
        if result["success"]:
            print("\n🎉 States to probabilities conversion completed successfully!")
            exit_code = 0
        else:
            print("\n❌ States to probabilities conversion failed!")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Conversion interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        exit_code = 1
    
    # Clean up PID file if created
    pid_file = "states_to_probabilities.pid"
    if os.path.exists(pid_file):
        try:
            os.remove(pid_file)
        except:
            pass
    
    exit(exit_code)
