"""
Crash-Safe Logging Decorator Module

This module provides decorators and utilities for crash-safe logging with separate process logging,
signal handling, heartbeat monitoring, and comprehensive error capture.

Usage:
    from crash_safe_logging import crash_safe_log, setup_logging

    @crash_safe_log()
    def my_function():
        # Your code here
        pass

    # Or for manual setup:
    @crash_safe_log(heartbeat_interval=5, log_level=logging.DEBUG)
    def my_function():
        # Your code here
        pass
"""

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
import functools
from datetime import datetime, timedelta
from typing import Optional, Callable, Any


class CrashSafeLogger:
    """
    A comprehensive crash-safe logging system that runs logging in a separate process.
    """
    
    def __init__(self, log_file_prefix: str = "execution", heartbeat_interval: float = 10.0, 
                 log_level: int = logging.DEBUG):
        self.log_file_prefix = log_file_prefix
        self.heartbeat_interval = heartbeat_interval
        self.log_level = log_level
        self.logger = None
        self.log_queue = None
        self.log_process = None
        self.log_file = None
        self.shutdown_event = None
        self.heartbeat_thread = None
        self._setup_complete = False
    
    @staticmethod
    def logging_process(log_queue: multiprocessing.Queue, log_file: str, 
                       shutdown_event: multiprocessing.Event):
        """
        Separate process for logging that runs concurrently with main code.
        This ensures logging continues even if main process crashes.
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
    
    def setup(self) -> logging.Logger:
        """
        Set up crash-safe logging with separate process.
        Returns the configured logger.
        """
        if self._setup_complete:
            return self.logger
            
        # Create organized log directory structure
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")  # YYYY-MM-DD format
        time_str = now.strftime("%H-%M-%S")  # HH-MM-SS format (24h)
        
        # Create logs directory structure: logs/YYYY-MM-DD/
        logs_dir = os.path.join("logs", date_str)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create log file with readable time format
        self.log_file = os.path.join(logs_dir, f"{self.log_file_prefix}_{time_str}.log")
        
        # Create queue for inter-process communication
        self.log_queue = multiprocessing.Queue()
        
        # Create shutdown event for graceful termination
        self.shutdown_event = multiprocessing.Event()
        
        # Start logging process
        self.log_process = multiprocessing.Process(
            target=self.logging_process, 
            args=(self.log_queue, self.log_file, self.shutdown_event)
        )
        self.log_process.start()
        
        # Give logging process time to start
        time.sleep(0.1)
        
        # Configure main process logger
        self.logger = logging.getLogger('MainProcess')
        self.logger.setLevel(self.log_level)
        
        # Create queue handler for main logger
        queue_handler = logging.handlers.QueueHandler(self.log_queue)
        self.logger.addHandler(queue_handler)
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        # Start heartbeat monitor
        self._start_heartbeat_monitor()
        
        self._setup_complete = True
        return self.logger
    
    def _setup_signal_handlers(self):
        """
        Set up signal handlers to catch interruptions and terminations.
        """
        def signal_handler(signum, frame):
            if self.logger:
                signal_name = signal.Signals(signum).name
                self.logger.critical(f"=== SIGNAL RECEIVED: {signal_name} ({signum}) ===")
                self.logger.critical(f"Signal received at frame: {frame}")
                self.logger.critical("Application is being terminated by signal")
                
                # Force immediate logging
                time.sleep(0.2)  # Give logger time to process
                
                # Signal shutdown to logging process
                self.shutdown_event.set()
                self.log_queue.put(None)
                
                # Exit with appropriate code
                sys.exit(128 + signum)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Set up atexit handler for other terminations
        def atexit_handler():
            if self.logger:
                self.logger.critical("=== ATEXIT HANDLER TRIGGERED ===")
                self.logger.critical("Application is terminating (atexit)")
                self.shutdown_event.set()
                time.sleep(0.2)  # Give logger time to process
        
        atexit.register(atexit_handler)
    
    def _start_heartbeat_monitor(self):
        """
        Create and start a heartbeat monitor that logs periodic status.
        """
        def heartbeat_worker():
            start_time = time.time()
            heartbeat_count = 0
            
            while not self.shutdown_event.is_set():
                try:
                    heartbeat_count += 1
                    elapsed = time.time() - start_time
                    self.logger.info(f"HEARTBEAT #{heartbeat_count} - Elapsed: {elapsed:.1f}s - Process alive")
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Heartbeat monitor error: {e}")
                    time.sleep(1)
        
        # Start heartbeat in a separate thread
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
    
    def log_system_info(self):
        """
        Log comprehensive system information.
        """
        if not self.logger:
            return
            
        self.logger.info("=== SYSTEM INFORMATION ===")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Log file path: {self.log_file}")
        self.logger.info(f"Log directory: {os.path.dirname(self.log_file)}")
        self.logger.info(f"Main process PID: {os.getpid()}")
        if self.log_process:
            self.logger.info(f"Logging process PID: {self.log_process.pid}")
        self.logger.info(f"Platform: {sys.platform}")
        self.logger.info(f"Heartbeat interval: {self.heartbeat_interval}s")
    
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with comprehensive error handling and logging.
        """
        if not self.logger:
            raise RuntimeError("Logger not set up. Call setup() first.")
            
        try:
            self.logger.info(f"Starting execution of {func.__name__}")
            result = func(*args, **kwargs)
            self.logger.info(f"Successfully completed {func.__name__}")
            return result
        except Exception as e:
            self.logger.error(f"ERROR in {func.__name__}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def cleanup(self):
        """
        Clean shutdown of the logging system.
        """
        if not self._setup_complete:
            return
            
        if self.logger:
            self.logger.info("=== BEGINNING SHUTDOWN SEQUENCE ===")
        
        # Stop heartbeat monitor
        if self.shutdown_event:
            self.shutdown_event.set()
        
        # Give some time for all log messages to be processed
        time.sleep(0.5)
        
        try:
            # Send sentinel to stop logging process
            if self.log_queue:
                self.log_queue.put(None)
            
            # Wait for logging process to finish with timeout
            if self.log_process:
                self.log_process.join(timeout=3)
                
                if self.log_process.is_alive():
                    print("Warning: Logging process did not terminate gracefully, forcing termination")
                    self.log_process.terminate()
                    self.log_process.join(timeout=2)
                    
                    if self.log_process.is_alive():
                        print("Error: Logging process still alive after terminate, killing forcefully")
                        self.log_process.kill()
                        self.log_process.join()
        except Exception as e:
            print(f"Error during logging cleanup: {e}")
        
        print(f"\nAll logs have been saved to: {self.log_file}")
        print(f"Log directory: {os.path.dirname(self.log_file)}")
        print("Check the log file for complete execution details, including any interruptions or crashes.")
        
        # Check if logging process exited abnormally
        if self.log_process and self.log_process.exitcode and self.log_process.exitcode != 0:
            print(f"WARNING: Logging process exited with code {self.log_process.exitcode}")
        
        # Final status check
        if self.log_process and self.log_process.is_alive():
            print("WARNING: Logging process is still running after cleanup")


def crash_safe_log(log_file_prefix: str = "execution", heartbeat_interval: float = 10.0, 
                   log_level: int = logging.DEBUG, log_system_info: bool = True):
    """
    Decorator that adds crash-safe logging to any function.
    
    Args:
        log_file_prefix: Prefix for the log file name
        heartbeat_interval: Interval in seconds for heartbeat messages
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_system_info: Whether to log system information at startup
    
    Usage:
        @crash_safe_log()
        def my_function():
            # Your code here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create crash-safe logger instance
            crash_logger = CrashSafeLogger(
                log_file_prefix=log_file_prefix,
                heartbeat_interval=heartbeat_interval,
                log_level=log_level
            )
            
            try:
                # Set up logging
                logger = crash_logger.setup()
                
                # Log function start
                logger.info(f"=== STARTING EXECUTION OF {func.__name__.upper()} ===")
                
                # Log system information if requested
                if log_system_info:
                    crash_logger.log_system_info()
                
                # Check if logging process is still alive
                if not crash_logger.log_process.is_alive():
                    raise RuntimeError("Logging process died unexpectedly")
                
                # Execute function with safety wrapper
                logger.info(f"About to start {func.__name__} - monitoring for crashes and interruptions")
                result = crash_logger.safe_execute(func, *args, **kwargs)
                
                logger.info(f"=== {func.__name__.upper()} COMPLETED SUCCESSFULLY ===")
                if hasattr(result, '__len__'):
                    logger.info(f"Result length: {len(result)}")
                
                return result
                
            except KeyboardInterrupt:
                if crash_logger.logger:
                    crash_logger.logger.critical("=== KEYBOARD INTERRUPT (Ctrl+C) DETECTED ===")
                    crash_logger.logger.critical("User manually interrupted the execution")
                    crash_logger.logger.critical("This interruption was successfully captured by the logging system")
                raise
            except SystemExit as e:
                if crash_logger.logger:
                    crash_logger.logger.critical(f"=== SYSTEM EXIT DETECTED ===")
                    crash_logger.logger.critical(f"Exit code: {e.code}")
                    crash_logger.logger.critical("System exit was captured by the logging system")
                raise
            except Exception as e:
                if crash_logger.logger:
                    crash_logger.logger.critical(f"=== CRITICAL ERROR - {func.__name__} execution failed ===")
                    crash_logger.logger.critical(f"Error: {str(e)}")
                    crash_logger.logger.critical(f"Full traceback: {traceback.format_exc()}")
                    crash_logger.logger.critical("This error was captured by the crash-safe logging system")
                    
                    # Check if the error might be due to external interruption
                    if any(keyword in str(e).lower() for keyword in ["abort", "interrupt", "control-c"]):
                        crash_logger.logger.critical("ERROR APPEARS TO BE RELATED TO EXTERNAL INTERRUPTION OR SIGNAL")
                
                raise
            finally:
                # Clean shutdown
                crash_logger.cleanup()
        
        return wrapper
    return decorator


def setup_logging(log_file_prefix: str = "execution", heartbeat_interval: float = 10.0, 
                  log_level: int = logging.DEBUG) -> tuple[logging.Logger, CrashSafeLogger]:
    """
    Set up crash-safe logging manually (alternative to using the decorator).
    
    Returns:
        tuple: (logger, crash_safe_logger_instance)
    
    Usage:
        logger, crash_logger = setup_logging()
        try:
            logger.info("Starting my process")
            # Your code here
        finally:
            crash_logger.cleanup()
    """
    crash_logger = CrashSafeLogger(
        log_file_prefix=log_file_prefix,
        heartbeat_interval=heartbeat_interval,
        log_level=log_level
    )
    
    logger = crash_logger.setup()
    return logger, crash_logger


def list_log_files(days_back: int = 7) -> dict:
    """
    List all log files from the last N days.
    
    Args:
        days_back: Number of days to look back for log files
    
    Returns:
        dict: Dictionary with date as key and list of log files as value
    """
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return {}
    
    log_files = {}
    now = datetime.now()
    
    for i in range(days_back):
        date = now - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        date_dir = os.path.join(logs_dir, date_str)
        
        if os.path.exists(date_dir):
            files = [f for f in os.listdir(date_dir) if f.endswith('.log')]
            if files:
                log_files[date_str] = sorted(files)
    
    return log_files


def get_latest_log_file() -> Optional[str]:
    """
    Get the path to the most recent log file.
    
    Returns:
        str: Path to the latest log file, or None if no logs found
    """
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return None
    
    latest_file = None
    latest_time = 0
    
    for date_dir in os.listdir(logs_dir):
        date_path = os.path.join(logs_dir, date_dir)
        if os.path.isdir(date_path):
            for log_file in os.listdir(date_path):
                if log_file.endswith('.log'):
                    file_path = os.path.join(date_path, log_file)
                    file_time = os.path.getmtime(file_path)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file_path
    
    return latest_file


def print_log_summary():
    """
    Print a summary of available log files organized by date.
    """
    log_files = list_log_files(30)  # Look back 30 days
    
    if not log_files:
        print("No log files found in the logs directory.")
        return
    
    print("=== LOG FILES SUMMARY ===")
    total_files = 0
    
    for date, files in sorted(log_files.items(), reverse=True):
        print(f"\n📅 {date} ({len(files)} files):")
        for file in files:
            file_path = os.path.join("logs", date, file)
            file_size = os.path.getsize(file_path)
            file_time = file.split('_')[-1].replace('.log', '').replace('-', ':')
            print(f"   ⏰ {file_time} - {file} ({file_size} bytes)")
            total_files += 1
    
    print(f"\nTotal log files: {total_files}")
    
    latest = get_latest_log_file()
    if latest:
        print(f"Latest log file: {latest}")


if __name__ == "__main__":
    # Example usage
    @crash_safe_log(log_file_prefix="test", heartbeat_interval=2.0)
    def test_function():
        """Test function to demonstrate the crash-safe logging decorator."""
        print("Running test function...")
        time.sleep(5)  # Simulate some work
        print("Test function completed!")
        return "success"
    
    # Run the test
    result = test_function()
    print(f"Function returned: {result}")
