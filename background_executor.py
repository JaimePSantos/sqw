"""
Background execution utilities for static noise quantum walk experiments.

This module handles safe background process execution with proper PID management
and logging for both Windows and Unix-like systems.
"""

import os
import sys
import time
import subprocess
from typing import Optional


def start_background_process(script_path: str, 
                           background_log_file: str, 
                           background_pid_file: str) -> bool:
    """
    Start the script in background mode safely.
    
    Args:
        script_path: Path to the script to run in background
        background_log_file: Log file for background process output
        background_pid_file: PID file to track background process
        
    Returns:
        True if background process started successfully, False otherwise
    """
    print("Starting SAFE background execution...")
    
    try:
        # Use sys.executable to get the current Python interpreter path
        python_executable = sys.executable
        
        # Create environment for subprocess that prevents recursion
        env = os.environ.copy()
        env['IS_BACKGROUND_PROCESS'] = '1'  # This prevents infinite recursion
        
        # Create log and PID file paths
        log_file_path = os.path.join(os.getcwd(), background_log_file)
        pid_file_path = os.path.join(os.getcwd(), background_pid_file)
        
        # Check if there's already a background process running
        if os.path.exists(pid_file_path):
            try:
                with open(pid_file_path, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if the old process is still running
                if _is_process_running(old_pid):
                    print(f"Background process already running (PID: {old_pid})")
                    if os.name == 'nt':  # Windows
                        print(f"   Kill it first with: taskkill /F /PID {old_pid}")
                    else:  # Unix-like
                        print(f"   Kill it first with: kill {old_pid}")
                    return False
                        
            except (ValueError, OSError):
                pass  # Invalid PID file, continue
        
        print("Starting background process...")
        
        # Initialize log file
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Background execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Command: {python_executable} {script_path}\n")
            log_file.write("=" * 50 + "\n\n")
        
        if os.name == 'nt':  # Windows - SAFE METHOD
            process = _start_windows_background(python_executable, script_path, env, log_file_path)
        else:  # Unix-like systems - SAFE METHOD
            process = _start_unix_background(python_executable, script_path, env, log_file_path)
        
        if process is None:
            print("Failed to start background process")
            return False
        
        # Save PID for cleanup
        with open(pid_file_path, 'w') as pid_file:
            pid_file.write(str(process.pid))
        
        # Give the process a moment to start and check if it's still running
        time.sleep(0.5)
        if process.poll() is None:
            print(f"Background process started safely (PID: {process.pid})")
            print(f"Monitor progress with: tail -f {log_file_path}")
            print(f"Stop with: kill {process.pid}")
            return True
        else:
            print(f"Warning: Background process (PID: {process.pid}) may have exited immediately")
            print(f"Check log file for details: {log_file_path}")
            return False
            
    except Exception as e:
        print(f"Error starting background process: {e}")
        return False


def _start_windows_background(python_executable: str, script_path: str, 
                            env: dict, log_file_path: str) -> Optional[subprocess.Popen]:
    """Start background process on Windows."""
    try:
        with open(log_file_path, 'a') as log_file:
            process = subprocess.Popen(
                [python_executable, "-u", script_path],  # -u for unbuffered output
                env=env,
                cwd=os.getcwd(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW  # Don't create visible window
            )
        return process
    except Exception as e:
        print(f"Failed to start Windows background process: {e}")
        return None


def _start_unix_background(python_executable: str, script_path: str, 
                         env: dict, log_file_path: str) -> Optional[subprocess.Popen]:
    """Start background process on Unix-like systems."""
    try:
        with open(log_file_path, 'a') as log_file:
            # First try with nohup and full detachment
            try:
                process = subprocess.Popen(
                    ["nohup", python_executable, "-u", script_path],
                    env=env,
                    cwd=os.getcwd(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # Create new session
                    start_new_session=True  # Additional detachment on Python 3.7+
                )
                return process
            except (TypeError, AttributeError, OSError) as e:
                print(f"   First attempt failed: {e}")
                # Fallback: Try without start_new_session
                try:
                    process = subprocess.Popen(
                        ["nohup", python_executable, "-u", script_path],
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid
                    )
                    return process
                except Exception as e2:
                    print(f"   Second attempt failed: {e2}")
                    # Final fallback: Basic subprocess without nohup
                    process = subprocess.Popen(
                        [python_executable, "-u", script_path],
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_file,
                        stderr=subprocess.STDOUT
                    )
                    return process
    except Exception as e:
        print(f"Failed to start Unix background process: {e}")
        return None


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is still running."""
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], 
                                  capture_output=True, text=True)
            return str(pid) in result.stdout
        else:  # Unix-like
            os.kill(pid, 0)  # Check if process exists
            return True
    except (OSError, subprocess.SubprocessError):
        return False


def cleanup_background_process(background_pid_file: str):
    """Clean up background process PID file."""
    try:
        if os.path.exists(background_pid_file):
            os.remove(background_pid_file)
            print("Cleaned up PID file")
    except Exception as e:
        print(f"Could not clean up PID file: {e}")


def is_background_process() -> bool:
    """Check if current process is running in background mode."""
    return os.environ.get('IS_BACKGROUND_PROCESS') == '1'


def setup_background_cleanup_handlers(background_log_file: str, background_pid_file: str):
    """Setup cleanup handlers for background process."""
    if not is_background_process():
        return
        
    import signal
    
    def cleanup_and_exit(signum, frame):
        try:
            with open(background_log_file, 'a') as f:
                f.write(f"\nReceived signal {signum}. Shutting down gracefully...\n")
        except:
            pass
        cleanup_background_process(background_pid_file)
        sys.exit(0)
    
    # Register signal handlers for background process
    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, cleanup_and_exit)  # Windows
