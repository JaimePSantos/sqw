"""
Logging utilities for multiprocessing quantum walk experiments.

This module provides centralized logging setup for both master and worker processes,
with appropriate formatting and file handling.
"""

import os
import logging
from typing import Tuple


def setup_process_logging(dev_value, process_id: int, log_dir: str = "process_logs") -> Tuple[logging.Logger, str]:
    """Setup logging for individual processes"""
    os.makedirs(log_dir, exist_ok=True)
    
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
    
    log_filename = os.path.join(log_dir, f"process_dev_{dev_str}_pid_{process_id}.log")
    
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


def setup_master_logging(log_filename: str = "static_experiment_multiprocess.log") -> Tuple[logging.Logger, str]:
    """Setup logging for the master process"""
    
    # Create master logger
    master_logger = logging.getLogger("master")
    master_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in master_logger.handlers[:]:
        master_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w')
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
    
    return master_logger, log_filename


def log_process_summary(logger: logging.Logger, process_results: list, process_info: dict):
    """Log summary of all process results."""
    logger.info("PROCESS SUMMARY:")
    successful_processes = 0
    failed_processes = 0
    
    for result in process_results:
        if result["success"]:
            successful_processes += 1
            logger.info(f"  SUCCESS: Dev {result['dev']} - PID {result['process_id']} - "
                       f"Samples: {result.get('computed_samples', 'N/A')} - "
                       f"Time: {result['total_time']:.1f}s - Log: {result['log_file']}")
        else:
            failed_processes += 1
            logger.error(f"  FAILED: Dev {result['dev']} - PID {result['process_id']} - "
                        f"Error: {result['error']} - Log: {result['log_file']}")
    
    logger.info(f"RESULTS: {successful_processes} successful, {failed_processes} failed processes")
    
    # Log process log file locations
    logger.info("PROCESS LOG FILES:")
    for dev, info in process_info.items():
        status = info.get('status', 'unknown')
        logger.info(f"  Dev {dev}: {info['log_file']} (Status: {status})")


def log_experiment_start(logger: logging.Logger, config):
    """Log experiment start information."""
    logger.info("=" * 60)
    logger.info("MULTIPROCESS STATIC NOISE EXPERIMENT STARTED")
    logger.info("=" * 60)
    logger.info(f"Parameters: N={config.N}, steps={config.steps}, samples={config.samples}")
    logger.info(f"Deviations: {config.devs}")
    logger.info(f"Max processes: {config.max_processes}")
    logger.info(f"Execution mode: {config.execution_mode}")


def log_experiment_completion(logger: logging.Logger, total_time: float, mode: str):
    """Log experiment completion."""
    logger.info("=" * 40)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 40)
    logger.info(f"Mode: {mode}")
    logger.info(f"Total execution time: {total_time:.2f} seconds")


def log_phase_start(logger: logging.Logger, phase_name: str):
    """Log the start of a major phase."""
    logger.info("=" * 40)
    logger.info(f"{phase_name.upper()} PHASE")
    logger.info("=" * 40)
