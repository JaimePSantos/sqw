#!/usr/bin/env python3

"""
Background Process Manager and Debugger

This script helps debug and manage background processes for the quantum walk experiments.
"""

import os
import sys
import subprocess
import time

def check_process_status(pid_file, log_file):
    """Check the status of a background process"""
    print(f"=== Checking Process Status ===")
    print(f"PID file: {pid_file}")
    print(f"Log file: {log_file}")
    
    # Check if PID file exists
    if not os.path.exists(pid_file):
        print("❌ No PID file found - no background process running")
        return False
    
    # Read PID
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        print(f"📋 PID from file: {pid}")
    except (ValueError, IOError) as e:
        print(f"❌ Error reading PID file: {e}")
        return False
    
    # Check if process is running
    is_running = False
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, shell=True
            )
            is_running = str(pid) in result.stdout
        else:  # Unix-like
            os.kill(pid, 0)  # Send signal 0 to check if process exists
            is_running = True
    except (OSError, subprocess.SubprocessError):
        is_running = False
    
    if is_running:
        print(f"✅ Process {pid} is running")
    else:
        print(f"❌ Process {pid} is not running")
        print("   Cleaning up stale PID file...")
        try:
            os.remove(pid_file)
            print("   ✅ Stale PID file removed")
        except OSError as e:
            print(f"   ❌ Could not remove PID file: {e}")
    
    # Check log file
    if os.path.exists(log_file):
        stat = os.stat(log_file)
        print(f"📄 Log file exists ({stat.st_size} bytes)")
        print(f"   Last modified: {time.ctime(stat.st_mtime)}")
        
        # Show last 10 lines
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    print("   Last 10 lines:")
                    for line in lines[-10:]:
                        print(f"     {line.rstrip()}")
                else:
                    print("   Log file is empty")
        except Exception as e:
            print(f"   Error reading log file: {e}")
    else:
        print("❌ No log file found")
    
    return is_running

def kill_background_process(pid_file):
    """Kill a background process"""
    if not os.path.exists(pid_file):
        print("❌ No PID file found - no background process to kill")
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
    except (ValueError, IOError) as e:
        print(f"❌ Error reading PID file: {e}")
        return False
    
    print(f"🔪 Attempting to kill process {pid}...")
    
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, text=True
            )
            success = result.returncode == 0
        else:  # Unix-like
            os.kill(pid, 15)  # SIGTERM
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if still alive
                os.kill(pid, 9)  # SIGKILL if still alive
                print("   Used SIGKILL (process was stubborn)")
            except OSError:
                pass  # Process is dead
            success = True
    except (OSError, subprocess.SubprocessError) as e:
        print(f"❌ Error killing process: {e}")
        success = False
    
    if success:
        print("✅ Process killed successfully")
        try:
            os.remove(pid_file)
            print("✅ PID file cleaned up")
        except OSError as e:
            print(f"⚠ Could not remove PID file: {e}")
    else:
        print("❌ Failed to kill process")
    
    return success

def monitor_log_file(log_file):
    """Monitor log file in real-time (like tail -f)"""
    if not os.path.exists(log_file):
        print(f"❌ Log file {log_file} does not exist")
        return
    
    print(f"📺 Monitoring {log_file} (Press Ctrl+C to stop)")
    
    try:
        with open(log_file, 'r') as f:
            # Go to end of file
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    print(line.rstrip())
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\\n📺 Monitoring stopped")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Background Process Manager")
        print()
        print("Usage:")
        print("  python bg_manager.py status [pid_file] [log_file]  - Check process status")
        print("  python bg_manager.py kill [pid_file]              - Kill background process")
        print("  python bg_manager.py monitor [log_file]           - Monitor log file")
        print("  python bg_manager.py test                         - Run diagnostics")
        print()
        print("Default files:")
        print("  PID file: static_experiment.pid")
        print("  Log file: static_experiment_background.log")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        pid_file = sys.argv[2] if len(sys.argv) > 2 else "static_experiment.pid"
        log_file = sys.argv[3] if len(sys.argv) > 3 else "static_experiment_background.log"
        check_process_status(pid_file, log_file)
        
    elif command == "kill":
        pid_file = sys.argv[2] if len(sys.argv) > 2 else "static_experiment.pid"
        kill_background_process(pid_file)
        
    elif command == "monitor":
        log_file = sys.argv[2] if len(sys.argv) > 2 else "static_experiment_background.log"
        monitor_log_file(log_file)
        
    elif command == "test":
        print("Running diagnostics...")
        print(f"Python executable: {sys.executable}")
        print(f"Platform: {os.name}")
        print(f"Current directory: {os.getcwd()}")
        
        # Test critical imports
        modules_to_test = [
            "numpy",
            "networkx", 
            "sqw.experiments_expanded_static",
            "jaime_scripts",
            "smart_loading_static"
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError as e:
                print(f"❌ {module}: {e}")
                
    else:
        print(f"Unknown command: {command}")
        print("Use 'python bg_manager.py' for help")

if __name__ == "__main__":
    main()
