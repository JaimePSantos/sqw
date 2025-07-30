from sqw.tesselations import even_cycle_two_tesselation
from sqw.experiments import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import logging.handlers
import multiprocessing
import queue
import threading
import traceback
import sys
import os
import signal
import atexit
import time
from datetime import datetime

def logging_process(log_queue, log_file, shutdown_event):
    """
    Separate process for logging that runs concurrently with main code.
    This ensures logging continues even if main process crashes.
    Uses shutdown_event for graceful termination.
    """
    # Configure logger for the logging process
    logger = logging.getLogger('CrashSafeLogger')
    logger.setLevel(logging.DEBUG)
    
    # Create file handler with immediate flush
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Set up signal handlers for the logging process
    def logging_signal_handler(signum, frame):
        logger.critical(f"LOGGING PROCESS: Received signal {signum} - forcing immediate shutdown")
        file_handler.flush()
        sys.exit(signum)
    
    signal.signal(signal.SIGINT, logging_signal_handler)
    signal.signal(signal.SIGTERM, logging_signal_handler)
    
    logger.info("=== LOGGING PROCESS STARTED ===")
    logger.info(f"Logging process PID: {os.getpid()}")
    
    try:
        while not shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                record = log_queue.get(timeout=0.5)
                if record is None:  # Sentinel to stop logging process
                    logger.info("Received shutdown sentinel")
                    break
                logger.handle(record)
                file_handler.flush()  # Force immediate write
            except queue.Empty:
                continue
            except Exception as e:
                # Log any errors in the logging process itself
                logger.error(f"Error in logging process: {e}")
                file_handler.flush()
    except KeyboardInterrupt:
        logger.critical("=== LOGGING PROCESS: KEYBOARD INTERRUPT DETECTED ===")
        file_handler.flush()
    except Exception as e:
        logger.critical(f"=== LOGGING PROCESS: UNEXPECTED ERROR: {e} ===")
        file_handler.flush()
    finally:
        logger.info("=== LOGGING PROCESS ENDED ===")
        file_handler.flush()
        file_handler.close()
        console_handler.close()

def setup_crash_safe_logging():
    """
    Set up crash-safe logging with separate process.
    Returns the log queue, process, and shutdown event for cleanup.
    """
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"test_interruption_{timestamp}.log"
    
    # Create queue for inter-process communication
    log_queue = multiprocessing.Queue()
    
    # Create shutdown event for graceful termination
    shutdown_event = multiprocessing.Event()
    
    # Start logging process
    log_process = multiprocessing.Process(target=logging_process, args=(log_queue, log_file, shutdown_event))
    log_process.start()
    
    # Give logging process time to start
    time.sleep(0.1)
    
    # Configure main process logger
    main_logger = logging.getLogger('MainProcess')
    main_logger.setLevel(logging.DEBUG)
    
    # Create queue handler for main logger
    queue_handler = logging.handlers.QueueHandler(log_queue)
    main_logger.addHandler(queue_handler)
    
    return main_logger, log_queue, log_process, log_file, shutdown_event

def setup_signal_handlers(logger, log_queue, shutdown_event):
    """
    Set up signal handlers to catch interruptions and terminations.
    """
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.critical(f"=== SIGNAL RECEIVED: {signal_name} ({signum}) ===")
        logger.critical(f"Signal received at frame: {frame}")
        logger.critical("Application is being terminated by signal")
        
        # Force immediate logging
        time.sleep(0.5)  # Give logger time to process
        
        # Signal shutdown to logging process
        shutdown_event.set()
        log_queue.put(None)
        
        # Wait a bit more for logging to complete
        time.sleep(0.5)
        
        # Exit with appropriate code
        sys.exit(128 + signum)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request

def test_long_running_task():
    """
    Simulate a long-running task that can be interrupted.
    """
    print("Starting long-running task. Will automatically stop after 5 seconds...")
    for i in range(5):
        time.sleep(1)  # Sleep for 1 second each iteration
        print(f"Working... {i+1}/5")
    print("Task completed normally")

if __name__ == "__main__":
    # Set up crash-safe logging
    logger, log_queue, log_process, log_file, shutdown_event = setup_crash_safe_logging()
    
    # Set up signal handlers for interruption detection
    setup_signal_handlers(logger, log_queue, shutdown_event)
    
    try:
        logger.info("=== STARTING INTERRUPTION TEST ===")
        logger.info(f"Main process PID: {os.getpid()}")
        logger.info(f"Logging process PID: {log_process.pid}")
        logger.info(f"Log file: {log_file}")
        
        test_long_running_task()
        
        logger.info("=== TEST COMPLETED SUCCESSFULLY ===")
        
    except KeyboardInterrupt:
        logger.critical("=== KEYBOARD INTERRUPT (Ctrl+C) DETECTED IN MAIN EXCEPT ===")
        logger.critical("User manually interrupted the execution")
        logger.critical("This interruption was successfully captured by the logging system")
    except SystemExit as e:
        logger.critical(f"=== SYSTEM EXIT DETECTED ===")
        logger.critical(f"Exit code: {e.code}")
        logger.critical("System exit was captured by the logging system")
    except Exception as e:
        logger.critical(f"=== CRITICAL ERROR ===")
        logger.critical(f"Error: {str(e)}")
        logger.critical(f"Full traceback: {traceback.format_exc()}")
    finally:
        # Clean shutdown of logging process
        logger.info("=== BEGINNING SHUTDOWN SEQUENCE ===")
        try:
            # Signal shutdown to logging process
            shutdown_event.set()
            # Send sentinel to stop logging process
            log_queue.put(None)
            
            # Wait for logging process to finish with timeout
            log_process.join(timeout=3)
            
            if log_process.is_alive():
                logger.warning("Logging process did not terminate gracefully, forcing termination")
                log_process.terminate()
                log_process.join(timeout=2)
        except Exception as e:
            print(f"Error during logging cleanup: {e}")
        
        print(f"\nAll logs have been saved to: {log_file}")
        print("Check the log file for complete execution details, including any interruptions.")
