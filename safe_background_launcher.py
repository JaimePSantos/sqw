#!/usr/bin/env python3

"""
Simplified background launcher for static_cluster_logged.py

This script provides a more robust way to run the main script in background
without relying on os.setsid which can cause issues on some systems.
"""

import os
import sys
import subprocess
import time

def launch_background():
    """Launch the main script in background with better compatibility"""
    
    # Get the main script path
    main_script = "static_cluster_logged.py"
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found in current directory")
        return False
    
    # Use current Python executable
    python_exe = sys.executable
    log_file = "static_experiment_background.log"
    pid_file = "static_experiment.pid"
    
    print("Starting background execution with simplified launcher...")
    print(f"Python: {python_exe}")
    print(f"Script: {main_script}")
    print(f"Log: {log_file}")
    
    # Check if already running
    if os.path.exists(pid_file):
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if process is still running
            try:
                if os.name == 'nt':  # Windows
                    result = subprocess.run(["tasklist", "/FI", f"PID eq {old_pid}"], 
                                          capture_output=True, text=True)
                    if str(old_pid) in result.stdout:
                        print(f"Background process already running (PID: {old_pid})")
                        return False
                else:  # Unix-like
                    os.kill(old_pid, 0)
                    print(f"Background process already running (PID: {old_pid})")
                    return False
            except OSError:
                pass  # Process doesn't exist
        except (ValueError, IOError):
            pass  # Invalid PID file
    
    # Prepare environment to prevent recursion
    env = os.environ.copy()
    env['IS_BACKGROUND_PROCESS'] = '1'
    env['RUN_IN_BACKGROUND'] = 'False'  # Disable the script's own background logic
    
    try:
        # Initialize log file
        with open(log_file, 'w') as f:
            f.write(f"Background execution started at {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Command: {python_exe} {main_script}\\n")
            f.write("=" * 50 + "\\n\\n")
        
        # Launch process with simplified approach
        with open(log_file, 'a') as log_f:
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    [python_exe, "-u", main_script],
                    env=env,
                    cwd=os.getcwd(),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:  # Unix-like - simplified approach
                # Try with nohup first, fallback to simple execution
                try:
                    process = subprocess.Popen(
                        ["nohup", python_exe, "-u", main_script],
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_f,
                        stderr=subprocess.STDOUT
                    )
                except (OSError, FileNotFoundError):
                    # Fallback: run without nohup
                    print("   nohup not available, using simple background execution")
                    process = subprocess.Popen(
                        [python_exe, "-u", main_script],
                        env=env,
                        cwd=os.getcwd(),
                        stdout=log_f,
                        stderr=subprocess.STDOUT
                    )
        
        # Save PID
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        # Check if process started successfully
        time.sleep(1)
        if process.poll() is None:
            print(f"✓ Background process started successfully (PID: {process.pid})")
            print(f"✓ Output logged to: {log_file}")
            print(f"✓ PID saved to: {pid_file}")
            
            if os.name == 'nt':
                print(f"\\n📺 Monitor with: Get-Content {log_file} -Wait")
                print(f"🔪 Kill with: taskkill /F /PID {process.pid}")
            else:
                print(f"\\n📺 Monitor with: tail -f {log_file}")
                print(f"🔪 Kill with: kill {process.pid}")
            
            return True
        else:
            print(f"✗ Background process exited immediately (code: {process.returncode})")
            print(f"Check {log_file} for details")
            return False
            
    except Exception as e:
        print(f"Error launching background process: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--no-background":
        print("Running in foreground mode...")
        # Set environment to disable background execution in main script
        os.environ['RUN_IN_BACKGROUND'] = 'False'
        # Run main script directly
        import static_cluster_logged
        return
    
    success = launch_background()
    if not success:
        print("\\nFalling back to foreground execution...")
        os.environ['RUN_IN_BACKGROUND'] = 'False'
        import static_cluster_logged

if __name__ == "__main__":
    main()
