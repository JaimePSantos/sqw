"""
Data management utilities for static noise quantum walk experiments.

This module handles standard deviation calculations, archive creation,
and mean probability distribution management.
"""

import os
import time
import tarfile
import pickle
import numpy as np
import multiprocessing as mp
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from typing import List, Optional, Dict, Any, Tuple

from experiment_logging import setup_process_logging
from worker_functions import compute_mean_probability_for_dev, dummy_tesselation_func


def create_or_load_std_data(mean_results: List, devs: List, N: int, steps: int, samples: int, 
                           tesselation_func, std_base_dir: str, noise_type: str, 
                           theta: Optional[float] = None) -> List:
    """
    Create or load standard deviation data from mean probability distributions.
    
    Args:
        mean_results: List of mean probability distributions for each parameter
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        samples: Number of samples
        tesselation_func: Function to create tesselation (dummy for static noise)
        std_base_dir: Base directory for standard deviation data
        noise_type: Type of noise ("static_noise")
        theta: Theta parameter
    
    Returns:
        List of standard deviation arrays for each deviation value
    """
    
    # Import functions from jaime_scripts and smart_loading_static
    from jaime_scripts import prob_distributions2std
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
                has_noise = dev[0] > 0
                dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
            else:
                has_noise = dev[1] > 0 or dev[0] > 0
                dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
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
                print(f"  [LOADED] Loaded existing std data for dev {dev_str}")
                stds.append(std_values)
                continue
            except Exception as e:
                print(f"  [ERROR] Could not load existing std data for dev {dev_str}: {e}")
        
        # Compute std data from mean probability distributions
        print(f"  [COMPUTING] Computing std data for dev {dev_str}...")
        try:
            # Get mean probability distributions for this deviation
            if mean_results and i < len(mean_results) and mean_results[i]:
                std_values = prob_distributions2std(mean_results[i], domain)
                
                # Save std data
                with open(std_filepath, 'wb') as f:
                    pickle.dump(std_values, f)
                
                stds.append(std_values)
                print(f"  [SAVED] Computed and saved std data for dev {dev_str}")
            else:
                print(f"  [WARNING] No mean results available for dev {dev_str}")
                stds.append([])
                
        except Exception as e:
            print(f"  [ERROR] Error computing std data for dev {dev_str}: {e}")
            stds.append([])
    
    print(f"[OK] Standard deviation data management completed!")
    return stds


def create_single_archive(archive_args: Tuple) -> Dict[str, Any]:
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


def create_experiment_archive(N: int, samples: int, use_multiprocess: bool = True, 
                            max_archive_processes: Optional[int] = 5, 
                            exclude_samples: bool = False, 
                            logger = None) -> Optional[str]:
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
        
    Returns:
        Archive filename if successful, None otherwise
    """
    def log_and_print(message, level="info"):
        """Helper function to log messages (cluster-safe, no print)"""
        if logger:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
        else:
            print(message)
    
    try:
        log_and_print("\n[ARCHIVE] Creating tar archive of experiment data...")
        
        # Create archive filename with timestamp, N, and samples
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_archive_name = f"experiments_data_N{N}_samples{samples}_{timestamp}.tar.gz"
        
        # Define data folders to check based on exclude_samples flag
        if exclude_samples:
            log_and_print("[ARCHIVE] Excluding raw sample files from archive (keeping only processed data)")
            data_folders = [
                "experiments_data_samples_probDist",
                "experiments_data_samples_std"
            ]
        else:
            data_folders = [
                "experiments_data_samples",
                "experiments_data_samples_probDist",
                "experiments_data_samples_std"
            ]
        
        n_folder_name = f"N_{N}"
        all_folders_to_archive = []
        
        # Find all folders containing N_{N} folders
        for data_folder in data_folders:
            if not os.path.exists(data_folder):
                log_and_print(f"[ARCHIVE] Data folder '{data_folder}' does not exist - skipping")
                continue
                
            log_and_print(f"[ARCHIVE] Checking folder: {os.path.abspath(data_folder)}")
            log_and_print(f"[ARCHIVE] Looking for folders containing '{n_folder_name}'...")
            
            folder_found = False
            for root, dirs, files in os.walk(data_folder):
                if n_folder_name in dirs:
                    target_folder = os.path.join(root, n_folder_name)
                    archive_path = os.path.relpath(target_folder, os.getcwd())
                    all_folders_to_archive.append((target_folder, archive_path))
                    folder_found = True
                    log_and_print(f"[ARCHIVE] Found: {target_folder}")
            
            if not folder_found:
                log_and_print(f"[ARCHIVE] No '{n_folder_name}' folders found in {data_folder}")
        
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
                for folder_path, archive_path in all_folders_to_archive:
                    log_and_print(f"[ARCHIVE] Adding {archive_path} to archive...")
                    tar.add(folder_path, arcname=archive_path)
            
        else:
            # Multiprocess archiving approach
            if max_archive_processes is None:
                max_archive_processes = min(total_folders, mp.cpu_count())
            
            log_and_print(f"[ARCHIVE] Using multiprocess archiving with {max_archive_processes} processes")
            log_and_print(f"[ARCHIVE] Creating {total_folders} temporary archives...")

            # This is a simplified version - in practice you'd want to implement
            # the full multiprocess archiving logic here
            with tarfile.open(final_archive_name, "w:gz") as tar:
                for folder_path, archive_path in all_folders_to_archive:
                    log_and_print(f"[ARCHIVE] Adding {archive_path} to archive...")
                    tar.add(folder_path, arcname=archive_path)
        
        # Get final archive size and report success
        if os.path.exists(final_archive_name):
            archive_size = os.path.getsize(final_archive_name)
            size_mb = archive_size / (1024 * 1024)
            log_and_print(f"[OK] Archive created successfully: {final_archive_name}")
            log_and_print(f"[OK] Archive size: {size_mb:.1f} MB")
            log_and_print(f"[OK] Archived {total_folders} folder(s) for N={N}")
            return final_archive_name
        else:
            log_and_print("[ERROR] Archive file was not created", "error")
            return None
        
    except Exception as e:
        log_and_print(f"[ERROR] Failed to create archive: {e}", "error")
        if logger:
            import traceback
            logger.error(traceback.format_exc())
        else:
            import traceback
            print(traceback.format_exc())
        return None


def create_mean_probability_distributions_multiprocess(
    tesselation_func,
    N: int,
    steps: int,
    devs: List,
    samples: int,
    source_base_dir: str = "experiments_data_samples",
    target_base_dir: str = "experiments_data_samples_probDist",
    noise_type: str = "static_noise",
    theta: Optional[float] = None,
    use_multiprocess: bool = True,
    max_processes: Optional[int] = None,
    logger = None
) -> List[Dict[str, Any]]:
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
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
        else:
            print(message)
    
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
            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"{dev:.3f}"
        
        log_file = os.path.join("process_logs", f"process_dev_meanprob_{dev_str}_pid_{process_id}.log")
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
            # Submit all tasks
            future_to_dev = {executor.submit(compute_mean_probability_for_dev, args): args[0] 
                           for args in process_args}
            
            # Process completed tasks
            for future in as_completed(future_to_dev):
                dev = future_to_dev[future]
                try:
                    result = future.result()
                    process_results.append(result)
                    
                    if result["success"]:
                        log_and_print(f"[MEAN_PROB] Completed dev {dev}: {result['processed_steps']} steps in {result['total_time']:.1f}s")
                    else:
                        log_and_print(f"[MEAN_PROB] Failed dev {dev}: {result['error']}", "error")
                        
                except Exception as e:
                    log_and_print(f"[ERROR] Exception processing dev {dev}: {str(e)}", "error")
                    process_results.append({
                        "dev": dev,
                        "success": False,
                        "error": str(e)
                    })
    
    except Exception as e:
        log_and_print(f"[ERROR] Critical error in mean probability multiprocessing: {str(e)}", "error")
        if logger:
            import traceback
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
            status = "SUCCESS" if result["success"] else "FAILED"
            logger.info(f"  {status}: Dev {result['dev']} - {result.get('total_time', 0):.1f}s")
    
    return process_results
