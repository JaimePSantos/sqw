"""
Signal handling and system monitoring utilities for multiprocessing experiments.

This module provides graceful shutdown handling, system resource monitoring,
and process management utilities.
"""

import os
import signal
import time
import psutil
import logging
from typing import Dict, Any, Optional

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\n[SHUTDOWN] Received signal {signum}. Initiating graceful shutdown...")
    print("[SHUTDOWN] Waiting for current processes to complete...")
    print("[SHUTDOWN] This may take a few minutes. Do not force-kill unless necessary.")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)  # Hangup signal (Unix)


def monitor_system_resources() -> Dict[str, float]:
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


def log_system_resources(logger: Optional[logging.Logger] = None, prefix: str = "[SYSTEM]"):
    """Log current system resource usage"""
    try:
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


def log_progress_update(phase: str, completed: int, total: int, start_time: float, 
                       logger: Optional[logging.Logger] = None):
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


def log_resource_usage(logger: logging.Logger, prefix: str = ""):
    """Log current resource usage"""
    try:
        resources = monitor_system_resources()
        logger.info(f"{prefix}Resource usage: Memory {resources['memory_percent']:.1f}%, "
                   f"Available {resources['memory_available_gb']:.1f}GB, "
                   f"CPU {resources['cpu_percent']:.1f}%")
    except Exception as e:
        logger.warning(f"Could not monitor resources: {e}")


def check_sample_exists(exp_dir: str, sample_id: int) -> bool:
    """Check if a specific sample already exists"""
    sample_file = os.path.join(exp_dir, f"sample_{sample_id}.pkl")
    return os.path.exists(sample_file)


def get_completed_samples(exp_dir: str, total_samples: int) -> list:
    """Get list of completed samples to enable resuming"""
    completed = []
    for i in range(total_samples):
        if check_sample_exists(exp_dir, i):
            completed.append(i)
    return completed


def print_resource_estimates(estimates: Dict[str, Any]):
    """Print computational resource estimates."""
    print(f"[COMPUTATION SCALE] Estimates based on configuration:")
    print(f"[RESOURCE ESTIMATE] Total quantum walks: {estimates['total_qw_simulations']}")
    print(f"[RESOURCE ESTIMATE] Estimated time: {estimates['estimated_time_minutes']:.1f} minutes")
    print(f"[MEMORY ESTIMATE] Per process: ~{estimates['memory_per_process_mb']:.1f}MB (streaming approach)")
    print(f"[MEMORY ESTIMATE] Total (all processes): ~{estimates['total_memory_mb']:.1f}MB")
    print(f"[TIMEOUT] Sample process timeout: {estimates['process_timeout_seconds']} seconds ({estimates['process_timeout_seconds']/3600:.1f} hours)")
    print(f"[TIMEOUT] Mean prob process timeout: {estimates['mean_prob_timeout_seconds']} seconds ({estimates['mean_prob_timeout_seconds']/3600:.1f} hours)")
    
    if estimates['total_memory_mb'] > 8000:
        print("[WARNING] High memory usage predicted!")
        print("   Consider reducing N or max_processes")
    else:
        print("[OK] Memory usage looks reasonable with streaming approach")


def print_system_info():
    """Print initial system information."""
    try:
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        print(f"[SYSTEM] Available memory: {memory.available / (1024**3):.1f}GB")
        print(f"[SYSTEM] CPU count: {cpu_count}")
    except ImportError:
        print("[SYSTEM] psutil not available - install for resource monitoring")
