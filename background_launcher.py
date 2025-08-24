#!/usr/bin/env python3
"""
Background launcher for static_cluster_logged_mp.py
This script properly launches the quantum walk experiment in background mode
to survive terminal disconnection on Linux clusters.
"""

import os
import sys
import signal
import subprocess
import time
from pathlib import Path

def launch_background():
    """Launch the experiment in true background mode"""
    
    # Set environment variable to indicate background mode
    env = os.environ.copy()
    env['BACKGROUND_MODE'] = '1'
    env['KEEP_TERMINAL_OUTPUT'] = '1'  # Keep initial output for confirmation
    
    print("=== Background Launcher for Quantum Walk Experiments ===")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Path to the main script
    script_path = Path(__file__).parent / "static_cluster_logged_mp.py"
    if not script_path.exists():
        print(f"ERROR: Could not find {script_path}")
        sys.exit(1)
    
    print(f"Script path: {script_path}")
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Launch the process
    print("\nLaunching experiment in background...")
    print("The process will continue running even if you disconnect from the terminal.")
    print("Use 'tail -f logs/master.log' to monitor progress.")
    print("Use 'ps aux | grep static_cluster' to check if the process is running.")
    
    try:
        # Use nohup-like behavior: redirect stdin from /dev/null and ignore SIGHUP
        with open('/dev/null', 'r') as devnull:
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdin=devnull,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # Detach from terminal session
                preexec_fn=os.setsid     # Create new process group
            )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's still running
        if process.poll() is None:
            print(f"\n✓ Process launched successfully with PID: {process.pid}")
            print(f"✓ The experiment is now running in the background")
            print(f"✓ You can safely disconnect from the terminal")
            
            # Write PID to file for monitoring
            with open("background_process.pid", "w") as f:
                f.write(str(process.pid))
            
            print(f"\nMonitoring commands:")
            print(f"  View logs: tail -f logs/master.log")
            print(f"  Check status: ps -p {process.pid}")
            print(f"  Kill process: kill {process.pid}")
            
        else:
            print(f"\n✗ Process exited immediately with code: {process.returncode}")
            # Try to get any error output
            try:
                output, _ = process.communicate(timeout=1)
                if output:
                    print("Output:")
                    print(output.decode())
            except:
                pass
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Failed to launch process: {e}")
        sys.exit(1)

def check_status():
    """Check if a background process is running"""
    pid_file = Path("background_process.pid")
    if not pid_file.exists():
        print("No background process PID file found")
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Signal 0 just checks if process exists
            print(f"Background process is running with PID: {pid}")
            return True
        except OSError:
            print(f"Background process with PID {pid} is not running")
            pid_file.unlink()  # Remove stale PID file
            return False
            
    except Exception as e:
        print(f"Error checking process status: {e}")
        return False

def kill_background():
    """Kill the background process"""
    pid_file = Path("background_process.pid")
    if not pid_file.exists():
        print("No background process PID file found")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"Killing background process with PID: {pid}")
        os.kill(pid, signal.SIGTERM)
        
        # Wait a bit for graceful shutdown
        time.sleep(2)
        
        # Check if it's still running
        try:
            os.kill(pid, 0)
            print("Process still running, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
        except OSError:
            print("Process terminated successfully")
        
        # Remove PID file
        pid_file.unlink()
        
    except Exception as e:
        print(f"Error killing process: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "status":
            check_status()
        elif command == "kill":
            kill_background()
        elif command == "launch":
            launch_background()
        else:
            print("Usage: python background_launcher.py [launch|status|kill]")
    else:
        # Default action is to launch
        launch_background()
