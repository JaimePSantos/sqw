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
from .config import get_config, get_advanced_config

# Try to import psutil, but handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Some system monitoring features will be disabled.")
    print("Install with: pip install psutil")

# Try to import resource module (not available on Windows)
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False
    # Create mock resource constants for Windows
    class MockResource:
        RLIM_INFINITY = -1
        RLIMIT_AS = 'RLIMIT_AS'
        RLIMIT_CPU = 'RLIMIT_CPU'
        RLIMIT_FSIZE = 'RLIMIT_FSIZE'
        RLIMIT_NPROC = 'RLIMIT_NPROC'
        
        def getrlimit(self, resource_type):
            return (self.RLIM_INFINITY, self.RLIM_INFINITY)
    
    resource = MockResource()


class CrashSafeLogger:
    """
    A comprehensive crash-safe logging system that runs logging in a separate process.
    """
    
    def __init__(self, log_file_prefix: str = None, heartbeat_interval: float = None, 
                 log_level: int = None):
        self.log_file_prefix = log_file_prefix or get_config("log_file_prefix")
        self.heartbeat_interval = heartbeat_interval or get_config("heartbeat_interval")
        self.log_level = log_level or get_config("log_level")
        self.logger = None
        self.log_queue = None
        self.log_process = None
        self.log_file = None
        self.shutdown_event = None
        self.heartbeat_thread = None
        self.deadman_thread = None
        self._setup_complete = False
        self._last_heartbeat_time = None
    
    @staticmethod
    def logging_process(log_queue, log_file: str, shutdown_event):
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
        formatter = logging.Formatter(get_config("log_format"))
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
                    # Get message from queue with shorter timeout for responsiveness
                    record = log_queue.get(timeout=0.5)  # Shorter timeout
                    if record is None:  # Sentinel to stop logging process
                        logger.info("Received shutdown sentinel")
                        break
                    logger.handle(record)
                    file_handler.flush()  # Force immediate write
                except queue.Empty:
                    # Check shutdown event more frequently
                    if shutdown_event.is_set():
                        break
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
        date_str = now.strftime(get_config("date_format"))
        time_str = now.strftime(get_config("time_format"))
        
        # Create logs directory structure: logs/YYYY-MM-DD/
        logs_dir = os.path.join(get_config("logs_base_directory"), date_str)
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
        self._setup_enhanced_signal_handlers()
        
        # Start heartbeat monitor
        self._start_heartbeat_monitor()
        
        # Start deadman's switch monitor
        self._start_deadman_monitor()
        
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
        Create and start a heartbeat monitor that logs periodic status with resource monitoring.
        """
        def heartbeat_worker():
            start_time = time.time()
            heartbeat_count = 0
            last_memory_check = 0
            memory_check_interval = 5  # Check memory every 5 heartbeats
            
            while not self.shutdown_event.is_set():
                try:
                    heartbeat_count += 1
                    elapsed = time.time() - start_time
                    current_time = time.time()
                    
                    # Update last heartbeat time for deadman's switch
                    self._last_heartbeat_time = current_time
                    
                    # Basic heartbeat
                    self.logger.info(f"HEARTBEAT #{heartbeat_count} - Elapsed: {elapsed:.1f}s - Process alive")
                    
                    # Periodic resource monitoring
                    if heartbeat_count - last_memory_check >= memory_check_interval:
                        self._log_resource_status()
                        last_memory_check = heartbeat_count
                    
                    # Enhanced cluster monitoring - check for signs of impending termination
                    if heartbeat_count % 20 == 0:  # Every 20 heartbeats
                        self._check_cluster_health()
                    
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Heartbeat monitor error: {e}")
                    time.sleep(1)
        
        # Start heartbeat in a separate thread
        self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
        self.heartbeat_thread.start()
        
        # Initialize heartbeat time
        self._last_heartbeat_time = time.time()
    
    def _start_deadman_monitor(self):
        """
        Start a deadman's switch that monitors if the main process suddenly stops.
        This runs in the logging process and will detect if heartbeats stop coming.
        """
        def deadman_worker():
            """Worker that runs in logging process to detect main process death."""
            time.sleep(5)  # Give initial setup time
            missed_heartbeats = 0
            max_missed = 3  # Allow 3 missed heartbeats before declaring process dead
            
            while not self.shutdown_event.is_set():
                try:
                    current_time = time.time()
                    if self._last_heartbeat_time:
                        time_since_last = current_time - self._last_heartbeat_time
                        expected_interval = self.heartbeat_interval * 1.5  # Allow some margin
                        
                        if time_since_last > expected_interval:
                            missed_heartbeats += 1
                            if missed_heartbeats >= max_missed:
                                # Main process appears to be dead
                                if self.logger:
                                    self.logger.critical("=== DEADMAN'S SWITCH TRIGGERED ===")
                                    self.logger.critical(f"No heartbeat for {time_since_last:.1f}s (expected every {self.heartbeat_interval}s)")
                                    self.logger.critical("MAIN PROCESS APPEARS TO HAVE TERMINATED SUDDENLY")
                                    self.logger.critical("This indicates: OOM kill, SIGKILL, hardware failure, or cluster termination")
                                    self.logger.critical("Process likely terminated without opportunity to log shutdown")
                                break
                        else:
                            missed_heartbeats = 0  # Reset counter if heartbeat received
                    
                    time.sleep(self.heartbeat_interval / 2)  # Check more frequently than heartbeat
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Deadman monitor error: {e}")
                    time.sleep(1)
        
        # This needs to be in the logging process, not main process
        # We'll implement this differently - create a file-based deadman's switch
        self._setup_file_deadman_switch()
    
    def _setup_file_deadman_switch(self):
        """
        Set up a file-based deadman's switch that can detect process termination.
        """
        def file_deadman_worker():
            deadman_file = os.path.join(os.path.dirname(self.log_file), "process_alive.txt")
            
            while not self.shutdown_event.is_set():
                try:
                    # Write current timestamp to file
                    with open(deadman_file, 'w') as f:
                        f.write(f"{time.time()}\n{os.getpid()}\n")
                    
                    time.sleep(self.heartbeat_interval / 2)
                    
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"File deadman switch error: {e}")
                    time.sleep(1)
            
            # Clean up deadman file on normal shutdown
            try:
                if os.path.exists(deadman_file):
                    os.remove(deadman_file)
            except:
                pass
        
        self.deadman_thread = threading.Thread(target=file_deadman_worker, daemon=True)
        self.deadman_thread.start()
    
    def _check_cluster_health(self):
        """
        Check for signs that the cluster might be about to terminate the job.
        """
        try:
            # Check if parent process is still alive (in case job scheduler changes)
            parent_pid = os.getppid()
            if PSUTIL_AVAILABLE:
                try:
                    parent_process = psutil.Process(parent_pid)
                    parent_name = parent_process.name()
                    self.logger.debug(f"Parent process: PID {parent_pid}, name: {parent_name}")
                except psutil.NoSuchProcess:
                    self.logger.warning(f"Parent process {parent_pid} no longer exists - possible job termination")
            else:
                self.logger.debug(f"Parent PID: {parent_pid} (psutil unavailable for process info)")
            
            # Check system load
            if PSUTIL_AVAILABLE and hasattr(psutil, 'getloadavg'):
                try:
                    load_avg = psutil.getloadavg()
                    self.logger.debug(f"System load average: {load_avg}")
                except Exception:
                    pass
            
            # Check disk space in working directory
            if PSUTIL_AVAILABLE:
                try:
                    disk_usage = psutil.disk_usage(os.getcwd())
                    free_space_gb = disk_usage.free / (1024**3)
                    if free_space_gb < 1.0:  # Less than 1GB free
                        self.logger.warning(f"LOW DISK SPACE: {free_space_gb:.2f} GB free")
                except Exception:
                    pass
            
            # Check for cluster-specific warning files or signals
            warning_files = [
                '/tmp/slurm_job_warning',
                '/tmp/pbs_job_warning', 
                '.cluster_warning',
                'job_warning'
            ]
            for warning_file in warning_files:
                if os.path.exists(warning_file):
                    self.logger.critical(f"CLUSTER WARNING FILE DETECTED: {warning_file}")
                    try:
                        with open(warning_file, 'r') as f:
                            content = f.read().strip()
                            self.logger.critical(f"Warning content: {content}")
                    except Exception:
                        pass
                        
        except Exception as e:
            self.logger.debug(f"Error in cluster health check: {e}")
    
    def _setup_enhanced_signal_handlers(self):
        """
        Set up enhanced signal handlers including cluster-specific signals.
        """
        def signal_handler(signum, frame):
            if self.logger:
                signal_name = signal.Signals(signum).name if hasattr(signal.Signals, signum) else str(signum)
                self.logger.critical(f"=== SIGNAL RECEIVED: {signal_name} ({signum}) ===")
                self.logger.critical(f"Signal received at frame: {frame}")
                self.logger.critical("Application is being terminated by signal")
                
                # Log additional context for specific signals
                if signum == signal.SIGTERM:
                    self.logger.critical("SIGTERM received - likely cluster job termination or timeout")
                elif signum == signal.SIGKILL:
                    self.logger.critical("SIGKILL received - forced termination (cannot be caught)")
                elif signum == signal.SIGUSR1:
                    self.logger.critical("SIGUSR1 received - possible cluster warning signal")
                elif signum == signal.SIGUSR2:
                    self.logger.critical("SIGUSR2 received - possible cluster checkpoint signal")
                
                # Log current resource state
                self._log_resource_status()
                
                # Force immediate logging
                time.sleep(0.2)  # Give logger time to process
                
                # Signal shutdown to logging process
                self.shutdown_event.set()
                self.log_queue.put(None)
                
                # Exit with appropriate code
                sys.exit(128 + signum)
        
        # Register standard signal handlers
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        # Register cluster-specific signals if available
        cluster_signals = [
            ('SIGUSR1', 'User-defined signal 1 - often used by clusters'),
            ('SIGUSR2', 'User-defined signal 2 - often used by clusters'),
            ('SIGXCPU', 'CPU time limit exceeded'),
            ('SIGXFSZ', 'File size limit exceeded')
        ]
        
        for sig_name, description in cluster_signals:
            if hasattr(signal, sig_name):
                sig_num = getattr(signal, sig_name)
                try:
                    signal.signal(sig_num, signal_handler)
                    self.logger.debug(f"Registered handler for {sig_name}: {description}")
                except (OSError, ValueError) as e:
                    self.logger.debug(f"Could not register {sig_name}: {e}")
        
        # Set up atexit handler for other terminations
        def atexit_handler():
            if self.logger:
                self.logger.critical("=== ATEXIT HANDLER TRIGGERED ===")
                self.logger.critical("Application is terminating (atexit)")
                self.logger.critical("This could indicate: normal exit, exception, or signal not caught")
                self._log_resource_status()
                self.shutdown_event.set()
                time.sleep(0.2)  # Give logger time to process
        
        atexit.register(atexit_handler)
    
    def log_system_info(self):
        """
        Log comprehensive system information including cluster-specific diagnostics.
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
        
        # Log cluster/system-specific information
        self._log_cluster_diagnostics()
    
    def _log_cluster_diagnostics(self):
        """
        Log cluster-specific diagnostic information to help identify termination causes.
        """
        try:
            # Check for cluster environment variables
            cluster_vars = [
                'SLURM_JOB_ID', 'SLURM_JOB_NAME', 'SLURM_PROCID', 'SLURM_LOCALID',
                'PBS_JOBID', 'PBS_JOBNAME', 'TORQUE_JOBID',
                'JOB_ID', 'SGE_JOB_ID', 'LSB_JOBID'
            ]
            
            self.logger.info("=== CLUSTER ENVIRONMENT DIAGNOSTICS ===")
            for var in cluster_vars:
                value = os.environ.get(var)
                if value:
                    self.logger.info(f"{var}: {value}")
            
            # System resource information
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_info = process.memory_info()
                    
                    self.logger.info("=== RESOURCE INFORMATION ===")
                    self.logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
                    self.logger.info(f"Virtual memory: {memory_info.vms / 1024 / 1024:.1f} MB")
                    self.logger.info(f"CPU count: {psutil.cpu_count()}")
                    self.logger.info(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.1f} MB")
                    self.logger.info(f"Total memory: {psutil.virtual_memory().total / 1024 / 1024:.1f} MB")
                except Exception as e:
                    self.logger.warning(f"Error getting psutil information: {e}")
            else:
                self.logger.warning("psutil not available - limited resource monitoring")
            
            # Process resource limits (if available)
            if RESOURCE_AVAILABLE:
                try:
                    # Memory limits
                    mem_limit = resource.getrlimit(resource.RLIMIT_AS)
                    self.logger.info(f"Memory limit (AS): {mem_limit[0] if mem_limit[0] != resource.RLIM_INFINITY else 'unlimited'}")
                    
                    # CPU time limits
                    cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
                    self.logger.info(f"CPU time limit: {cpu_limit[0] if cpu_limit[0] != resource.RLIM_INFINITY else 'unlimited'} seconds")
                    
                    # File size limits
                    fsize_limit = resource.getrlimit(resource.RLIMIT_FSIZE)
                    self.logger.info(f"File size limit: {fsize_limit[0] if fsize_limit[0] != resource.RLIM_INFINITY else 'unlimited'} bytes")
                    
                    # Number of processes
                    nproc_limit = resource.getrlimit(resource.RLIMIT_NPROC)
                    self.logger.info(f"Process limit: {nproc_limit[0] if nproc_limit[0] != resource.RLIM_INFINITY else 'unlimited'}")
                    
                except (OSError, AttributeError) as e:
                    self.logger.warning(f"Could not get resource limits: {e}")
            else:
                self.logger.info("Resource limits: Not available on this platform (Windows)")
            
            # Check if we're in a container or virtualized environment
            if os.path.exists('/proc/1/cgroup'):
                with open('/proc/1/cgroup', 'r') as f:
                    cgroup_info = f.read()
                    if 'docker' in cgroup_info or 'lxc' in cgroup_info:
                        self.logger.info("Running in containerized environment")
            
            # Check for common cluster files
            cluster_files = [
                '/var/spool/slurm', '/var/spool/pbs', '/opt/sge',
                '/etc/slurm', '/etc/pbs', '/etc/torque'
            ]
            for cluster_file in cluster_files:
                if os.path.exists(cluster_file):
                    self.logger.info(f"Cluster system detected: {cluster_file}")
                    break
                    
        except Exception as e:
            self.logger.warning(f"Error gathering cluster diagnostics: {e}")
    
    def _log_resource_status(self):
        """
        Log current resource usage for monitoring resource exhaustion.
        """
        if not PSUTIL_AVAILABLE:
            self.logger.debug("psutil not available - skipping resource status")
            return
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Memory usage
            memory_mb = memory_info.rss / 1024 / 1024
            virtual_mb = memory_info.vms / 1024 / 1024
            
            # System memory
            sys_mem = psutil.virtual_memory()
            available_mb = sys_mem.available / 1024 / 1024
            mem_percent = sys_mem.percent
            
            self.logger.info(f"RESOURCES - Memory: {memory_mb:.1f}MB/{virtual_mb:.1f}MB virt, "
                           f"System: {mem_percent:.1f}% used ({available_mb:.1f}MB free), "
                           f"CPU: {cpu_percent:.1f}%")
                           
            # Check for concerning resource usage
            if memory_mb > 1000:  # > 1GB
                self.logger.warning(f"HIGH MEMORY USAGE: {memory_mb:.1f} MB")
            if mem_percent > 90:
                self.logger.warning(f"SYSTEM MEMORY CRITICAL: {mem_percent:.1f}% used")
            if cpu_percent > 95:
                self.logger.warning(f"HIGH CPU USAGE: {cpu_percent:.1f}%")
                
        except Exception as e:
            self.logger.warning(f"Error logging resource status: {e}")
    
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
        
        # Give minimal time for log messages to be processed
        time.sleep(0.1)
        
        try:
            # Send sentinel to stop logging process
            if self.log_queue:
                self.log_queue.put(None)
            
            # Wait for logging process to finish with short timeout
            if self.log_process:
                # Use shorter timeout to prevent hanging
                self.log_process.join(timeout=1)
                
                if self.log_process.is_alive():
                    print("Warning: Logging process did not terminate gracefully, forcing termination")
                    self.log_process.terminate()
                    # Don't wait long for termination
                    self.log_process.join(timeout=0.5)
                    
                    if self.log_process.is_alive():
                        print("Error: Logging process still alive after terminate, killing forcefully")
                        try:
                            self.log_process.kill()
                            self.log_process.join(timeout=0.5)
                        except:
                            # On Windows, sometimes kill() can fail, just continue
                            pass
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


def crash_safe_log(log_file_prefix: str = None, heartbeat_interval: float = None, 
                   log_level: int = None, log_system_info: bool = None):
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
            
            # Use config default for log_system_info if not specified
            should_log_system_info = log_system_info if log_system_info is not None else get_config("log_system_info")
            
            try:
                # Set up logging
                logger = crash_logger.setup()
                
                # Log function start
                logger.info(f"=== STARTING EXECUTION OF {func.__name__.upper()} ===")
                
                # Log system information if requested
                if should_log_system_info:
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


def setup_logging(log_file_prefix: str = None, heartbeat_interval: float = None, 
                  log_level: int = None) -> tuple[logging.Logger, CrashSafeLogger]:
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
    logs_dir = get_config("logs_base_directory")
    if not os.path.exists(logs_dir):
        return {}
    
    log_files = {}
    now = datetime.now()
    
    for i in range(days_back):
        date = now - timedelta(days=i)
        date_str = date.strftime(get_config("date_format"))
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
    logs_dir = get_config("logs_base_directory")
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
            file_path = os.path.join(get_config("logs_base_directory"), date, file)
            file_size = os.path.getsize(file_path)
            file_time = file.split('_')[-1].replace('.log', '').replace('-', ':')
            print(f"   ⏰ {file_time} - {file} ({file_size} bytes)")
            total_files += 1
    
    print(f"\nTotal log files: {total_files}")
    
    latest = get_latest_log_file()
    if latest:
        print(f"Latest log file: {latest}")


def check_for_crashed_processes():
    """
    Check for evidence of processes that crashed without proper shutdown.
    Looks for orphaned deadman switch files and provides diagnostics.
    """
    logs_dir = get_config("logs_base_directory")
    if not os.path.exists(logs_dir):
        print("No logs directory found.")
        return
    
    print("=== CRASH DETECTION ANALYSIS ===")
    found_crashes = False
    
    # Look for orphaned deadman files
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file == "process_alive.txt":
                deadman_file = os.path.join(root, file)
                try:
                    with open(deadman_file, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            timestamp = float(lines[0].strip())
                            pid = int(lines[1].strip())
                            
                            # Check if process is still running
                            if PSUTIL_AVAILABLE:
                                try:
                                    process = psutil.Process(pid)
                                    if process.is_running():
                                        continue  # Process still alive
                                except psutil.NoSuchProcess:
                                    pass  # Process is dead
                            else:
                                # Without psutil, we can't check if process is running
                                # Assume process is dead if file is old enough
                                pass
                            
                            # Check how long ago the process died
                            time_since = time.time() - timestamp
                            if time_since > 60:  # More than 1 minute ago
                                found_crashes = True
                                print(f"\n🚨 CRASHED PROCESS DETECTED:")
                                print(f"   Directory: {root}")
                                print(f"   PID: {pid}")
                                print(f"   Last seen: {time_since/60:.1f} minutes ago")
                                print(f"   Timestamp: {datetime.fromtimestamp(timestamp)}")
                                
                                # Look for corresponding log file
                                log_files = [f for f in os.listdir(root) if f.endswith('.log')]
                                if log_files:
                                    latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(root, x)))
                                    print(f"   Log file: {latest_log}")
                                    
                                    # Check log file for termination messages
                                    log_path = os.path.join(root, latest_log)
                                    with open(log_path, 'r') as log_f:
                                        log_content = log_f.read()
                                        if "SIGNAL RECEIVED" in log_content:
                                            print("   ✓ Signal termination logged")
                                        elif "KEYBOARD INTERRUPT" in log_content:
                                            print("   ✓ Keyboard interrupt logged")
                                        elif "SHUTDOWN SEQUENCE" in log_content:
                                            print("   ✓ Normal shutdown logged")
                                        else:
                                            print("   ❌ NO TERMINATION REASON LOGGED - LIKELY SUDDEN DEATH")
                                            if "HIGH MEMORY USAGE" in log_content:
                                                print("   📊 High memory usage detected - possible OOM kill")
                                            if "SYSTEM MEMORY CRITICAL" in log_content:
                                                print("   📊 Critical memory usage detected - likely OOM kill")
                                            
                                        # Extract last few log lines
                                        log_lines = log_content.strip().split('\n')
                                        print("   Last log entries:")
                                        for line in log_lines[-5:]:
                                            if line.strip():
                                                print(f"     {line}")
                                
                                # Clean up the deadman file
                                os.remove(deadman_file)
                                print(f"   🧹 Cleaned up deadman file")
                                
                except Exception as e:
                    print(f"Error analyzing {deadman_file}: {e}")
    
    if not found_crashes:
        print("✅ No evidence of crashed processes found.")
    else:
        print(f"\n💡 CRASH ANALYSIS COMPLETE")
        print("Common causes of sudden process termination:")
        print("  - OOM (Out of Memory) kills by the kernel")
        print("  - Resource limit violations (CPU time, memory, etc.)")
        print("  - Cluster scheduler job termination")
        print("  - Hardware failures or node crashes")
        print("  - SIGKILL signals (cannot be caught)")
        print("\nTo prevent crashes:")
        print("  - Monitor memory usage in logs")
        print("  - Check cluster job time limits")
        print("  - Use checkpointing for long-running jobs")
        print("  - Consider splitting large computations")


def generate_cluster_diagnostic_script(output_file: str = "cluster_diagnostics.sh"):
    """
    Generate a shell script to gather cluster diagnostic information.
    """
    script_content = '''#!/bin/bash
# Cluster Diagnostics Script
# Run this on your cluster to gather system information

echo "=== CLUSTER DIAGNOSTIC INFORMATION ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "User: $(whoami)"
echo

echo "=== CLUSTER ENVIRONMENT ==="
env | grep -E "(SLURM|PBS|TORQUE|SGE|LSB)" || echo "No cluster environment variables found"
echo

echo "=== RESOURCE LIMITS ==="
ulimit -a
echo

echo "=== MEMORY INFORMATION ==="
free -h
echo

echo "=== CPU INFORMATION ==="
lscpu | head -20
echo

echo "=== DISK SPACE ==="
df -h
echo

echo "=== SYSTEM LOAD ==="
uptime
echo

echo "=== PROCESS LIMITS ==="
cat /proc/sys/kernel/pid_max 2>/dev/null || echo "Cannot read pid_max"
echo

echo "=== CGROUP INFORMATION ==="
cat /proc/1/cgroup 2>/dev/null || echo "Cannot read cgroup info"
echo

echo "=== CLUSTER SPECIFIC FILES ==="
for dir in /var/spool/slurm /var/spool/pbs /opt/sge /etc/slurm /etc/pbs /etc/torque; do
    if [ -d "$dir" ]; then
        echo "Found cluster directory: $dir"
        ls -la "$dir" 2>/dev/null | head -10
    fi
done
echo

echo "=== SYSTEM MESSAGES (last 50 lines) ==="
sudo tail -50 /var/log/messages 2>/dev/null || \\
sudo tail -50 /var/log/syslog 2>/dev/null || \\
echo "Cannot access system logs (try: sudo journalctl -n 50)"
echo

echo "=== MEMORY PRESSURE INFORMATION ==="
cat /proc/pressure/memory 2>/dev/null || echo "Memory pressure info not available"
echo

echo "=== OOM KILLER LOGS ==="
sudo dmesg | grep -i "killed process\\|out of memory\\|oom" | tail -10 || echo "Cannot access dmesg"
echo

echo "=== END DIAGNOSTICS ==="
'''
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(output_file, 0o755)
    
    print(f"Generated cluster diagnostic script: {output_file}")
    print("Run this on your cluster with: bash cluster_diagnostics.sh")
    print("This will help identify cluster-specific termination causes.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crash-safe logging utilities")
    parser.add_argument("--check-crashes", action="store_true", 
                       help="Check for evidence of crashed processes")
    parser.add_argument("--generate-diagnostics", action="store_true",
                       help="Generate cluster diagnostic script")
    parser.add_argument("--test", action="store_true", help="Run test function")
    
    args = parser.parse_args()
    
    if args.check_crashes:
        check_for_crashed_processes()
    elif args.generate_diagnostics:
        generate_cluster_diagnostic_script()
    elif args.test:
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
    else:
        print("Use --help to see available options")
        print_log_summary()
