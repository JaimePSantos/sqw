#!/usr/bin/env python3
"""
Cluster logging additions - add these imports and functions to your existing script.
This provides comprehensive logging without breaking your existing code structure.
"""

import logging
import signal
import time
import traceback
import atexit
import threading
from datetime import datetime

# Add this at the top of your script, right after the imports
def setup_cluster_logging():
    """Setup comprehensive logging for cluster debugging."""
    log_filename = f"quantum_walk_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - PID:%(process)d - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('quantum_walk')
    logger.info(f"=== Quantum Walk Experiment Started ===")
    logger.info(f"Log file: {log_filename}")
    logger.info(f"Process ID: {os.getpid()}")
    
    # Log environment info
    env_vars = ['SSH_CLIENT', 'SSH_TTY', 'TMUX', 'STY', 'SLURM_JOB_ID', 'PBS_JOBID']
    for key in env_vars:
        if key in os.environ:
            logger.info(f"{key}: {os.environ[key]}")
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.error(f"!!! RECEIVED SIGNAL {signum} at {datetime.now()}")
        logger.error(f"Signal name: {signal.Signals(signum).name}")
        if signum == signal.SIGHUP:
            logger.error("SIGHUP: SSH connection likely closed!")
        elif signum == signal.SIGTERM:
            logger.error("SIGTERM: Process being terminated!")
        sys.exit(128 + signum)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)
    
    # Setup heartbeat
    def heartbeat():
        count = 0
        while True:
            time.sleep(300)  # Every 5 minutes
            count += 1
            logger.info(f">>> HEARTBEAT #{count}: Process alive at {datetime.now()}")
    
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    
    return logger

# Add this function to wrap your main experiment code
def log_experiment_progress(logger, func_name, *args, **kwargs):
    """Wrapper to log function entry/exit and catch exceptions."""
    logger.info(f"=== ENTERING {func_name} ===")
    start_time = time.time()
    
    try:
        # Call the original function
        if func_name == "main":
            from your_original_script import main
            result = main(*args, **kwargs)
        elif func_name == "run_experiment":
            from your_original_script import run_experiment
            result = run_experiment(*args, **kwargs)
        # Add other functions as needed
        
        elapsed = time.time() - start_time
        logger.info(f"=== {func_name} COMPLETED in {elapsed:.2f}s ===")
        return result
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"!!! EXCEPTION in {func_name} after {elapsed:.2f}s !!!")
        logger.error(f"Exception: {type(e).__name__}: {e}")
        logger.error("Traceback:")
        for line in traceback.format_exc().split('\n'):
            logger.error(f"  {line}")
        raise

# Example usage - add to your __main__ block:
"""
if __name__ == "__main__":
    import os
    import sys
    
    # Setup logging first thing
    logger = setup_cluster_logging()
    
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
            logger.info("Running in virtual environment mode")
            log_experiment_progress(logger, "run_experiment")
        else:
            logger.info("Running main setup function")
            log_experiment_progress(logger, "main")
            
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        sys.exit(1)
"""
