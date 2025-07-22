#!/usr/bin/env python3
"""
Debug wrapper for cluster execution - adds comprehensive logging to detect early termination.
Use this to wrap your existing script and capture all signals, exceptions, and execution flow.
"""

import sys
import os
import subprocess
import logging
import signal
import time
import traceback
import atexit
import threading
from datetime import datetime
from pathlib import Path

class ClusterDebugWrapper:
    def __init__(self, script_name):
        self.script_name = script_name
        self.setup_logging()
        self.setup_signal_handlers()
        self.setup_exit_handler()
        self.start_heartbeat()
        
    def setup_logging(self):
        """Setup comprehensive logging to file and console."""
        self.log_filename = f"cluster_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Configure logging with both file and console handlers
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - PID:%(process)d - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"=== Cluster Debug Wrapper Started ===")
        self.logger.info(f"Target script: {self.script_name}")
        self.logger.info(f"Log file: {self.log_filename}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info(f"Command line args: {sys.argv}")
        self.logger.info(f"Working directory: {os.getcwd()}")
        self.logger.info(f"Process ID: {os.getpid()}")
        self.logger.info(f"Parent Process ID: {os.getppid()}")
        
        # Log environment variables that might indicate session type
        env_vars = ['SHELL', 'TERM', 'SSH_CLIENT', 'SSH_TTY', 'DISPLAY', 'TMUX', 'STY', 
                   'SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'JOB_ID', 'HOSTNAME']
        self.logger.info(f"Environment variables:")
        for key in env_vars:
            value = os.environ.get(key, 'Not set')
            self.logger.info(f"  {key}: {value}")
        
        # Check if we're in a job scheduler environment
        job_schedulers = ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'JOB_ID']
        in_job = any(os.environ.get(var) for var in job_schedulers)
        self.logger.info(f"Running in job scheduler: {in_job}")
        
        # Check if SSH session
        ssh_session = any(os.environ.get(var) for var in ['SSH_CLIENT', 'SSH_TTY'])
        self.logger.info(f"SSH session detected: {ssh_session}")
        
    def setup_signal_handlers(self):
        """Setup signal handlers to catch termination signals."""
        
        def signal_handler(signum, frame):
            self.logger.error(f"!!! RECEIVED SIGNAL {signum} ({signal.Signals(signum).name}) !!!")
            self.logger.error(f"Signal received at: {datetime.now()}")
            self.logger.error(f"Frame info: {frame.f_code.co_filename}:{frame.f_lineno}")
            self.logger.error(f"Stack trace:")
            for line in traceback.format_stack(frame):
                self.logger.error(f"  {line.strip()}")
            
            # Log what this signal typically means
            signal_meanings = {
                signal.SIGTERM: "SIGTERM - Termination request (clean shutdown)",
                signal.SIGINT: "SIGINT - Interrupt signal (Ctrl+C)",
                signal.SIGHUP: "SIGHUP - Hangup signal (terminal closed/SSH disconnect)",
                signal.SIGKILL: "SIGKILL - Kill signal (cannot be caught)",
                signal.SIGQUIT: "SIGQUIT - Quit signal",
                signal.SIGUSR1: "SIGUSR1 - User-defined signal 1",
                signal.SIGUSR2: "SIGUSR2 - User-defined signal 2"
            }
            
            meaning = signal_meanings.get(signum, f"Unknown signal {signum}")
            self.logger.error(f"Signal meaning: {meaning}")
            
            if signum == signal.SIGHUP:
                self.logger.error("SIGHUP typically indicates:")
                self.logger.error("- SSH connection was closed")
                self.logger.error("- Terminal session ended")
                self.logger.error("- Process should have been started with nohup or screen/tmux")
                
            elif signum == signal.SIGTERM:
                self.logger.error("SIGTERM typically indicates:")
                self.logger.error("- Job scheduler timeout")
                self.logger.error("- System shutdown")
                self.logger.error("- Manual termination")
                
            # Log system resource usage
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                self.logger.error(f"Resource usage at termination:")
                self.logger.error(f"  User time: {usage.ru_utime:.2f}s")
                self.logger.error(f"  System time: {usage.ru_stime:.2f}s")
                self.logger.error(f"  Max memory: {usage.ru_maxrss / 1024:.2f} MB")
            except:
                pass
            
            self.logger.error(f"Process {os.getpid()} exiting due to signal {signum}")
            sys.exit(128 + signum)
        
        # Register handlers for common signals
        signals_to_catch = [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]
        
        # Add more signals if available on this platform
        for sig_name in ['SIGUSR1', 'SIGUSR2', 'SIGQUIT']:
            if hasattr(signal, sig_name):
                signals_to_catch.append(getattr(signal, sig_name))
        
        for sig in signals_to_catch:
            signal.signal(sig, signal_handler)
            self.logger.info(f"Signal handler registered for {signal.Signals(sig).name}")
    
    def setup_exit_handler(self):
        """Setup exit handler to log when process exits normally."""
        
        def exit_handler():
            self.logger.info("=== Process exiting normally ===")
            self.logger.info(f"Exit time: {datetime.now()}")
            
            # Log final resource usage
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                self.logger.info(f"Final resource usage:")
                self.logger.info(f"  User time: {usage.ru_utime:.2f}s")
                self.logger.info(f"  System time: {usage.ru_stime:.2f}s")
                self.logger.info(f"  Max memory: {usage.ru_maxrss / 1024:.2f} MB")
            except:
                pass
            
            self.logger.info("Normal exit - script completed successfully")
        
        atexit.register(exit_handler)
        self.logger.info("Exit handler registered")
    
    def start_heartbeat(self):
        """Start heartbeat thread to log periodic status."""
        
        def heartbeat():
            """Log a heartbeat message periodically."""
            count = 0
            while True:
                time.sleep(300)  # Every 5 minutes
                count += 1
                self.logger.info(f">>> HEARTBEAT #{count}: Process {os.getpid()} alive at {datetime.now()}")
                
                # Log memory usage periodically
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    self.logger.info(f">>> HEARTBEAT #{count}: Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
                except:
                    pass
        
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        self.logger.info("Heartbeat thread started - will log every 5 minutes")
    
    def run_script(self):
        """Run the target script with full logging."""
        self.logger.info(f"=== Starting execution of {self.script_name} ===")
        start_time = time.time()
        
        try:
            # Check if script exists
            if not os.path.exists(self.script_name):
                self.logger.error(f"Script not found: {self.script_name}")
                return 1
            
            # Log script details
            script_stat = os.stat(self.script_name)
            self.logger.info(f"Script size: {script_stat.st_size} bytes")
            self.logger.info(f"Script modified: {datetime.fromtimestamp(script_stat.st_mtime)}")
            
            # Execute the script
            self.logger.info("Executing script...")
            
            # Import and execute the script
            import importlib.util
            spec = importlib.util.spec_from_file_location("target_script", self.script_name)
            if spec is None:
                self.logger.error(f"Could not load script: {self.script_name}")
                return 1
                
            module = importlib.util.module_from_spec(spec)
            
            # Execute the script
            spec.loader.exec_module(module)
            
            elapsed = time.time() - start_time
            self.logger.info(f"=== Script execution completed successfully in {elapsed:.2f} seconds ===")
            return 0
            
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            self.logger.error(f"!!! Script interrupted by user after {elapsed:.2f} seconds !!!")
            return 130
            
        except SystemExit as e:
            elapsed = time.time() - start_time
            self.logger.error(f"!!! Script called sys.exit({e.code}) after {elapsed:.2f} seconds !!!")
            return e.code if e.code is not None else 0
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"!!! EXCEPTION in script execution after {elapsed:.2f} seconds !!!")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception message: {str(e)}")
            self.logger.error(f"Full traceback:")
            for line in traceback.format_exc().split('\n'):
                self.logger.error(f"  {line}")
            return 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster_debug_wrapper.py <script_to_run.py>")
        print("This will run your script with comprehensive logging to debug cluster issues.")
        sys.exit(1)
    
    script_name = sys.argv[1]
    
    # Remove wrapper script from sys.argv so target script sees correct arguments
    sys.argv = [script_name] + sys.argv[2:]
    
    wrapper = ClusterDebugWrapper(script_name)
    exit_code = wrapper.run_script()
    
    wrapper.logger.info(f"=== Cluster Debug Wrapper finished with exit code {exit_code} ===")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
