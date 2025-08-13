#!/usr/bin/env python3

"""
Background Process Manager for Static Experiment

This utility helps safely manage background processes created by static_cluster_logged.py
"""

import os
import sys
import subprocess
import signal

BACKGROUND_PID_FILE = "static_experiment.pid"
BACKGROUND_LOG_FILE = "static_experiment_background.log"

def get_process_pid():
    """Get the PID of the background process if it exists."""
    if os.path.exists(BACKGROUND_PID_FILE):
        try:
            with open(BACKGROUND_PID_FILE, 'r') as f:
                return int(f.read().strip())
        except (ValueError, IOError):
            return None
    return None

def is_process_running(pid):
    """Check if a process with given PID is running."""
    if pid is None:
        return False
    
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True, text=True, shell=True
            )
            return str(pid) in result.stdout
        else:  # Unix-like
            os.kill(pid, 0)  # Send signal 0 to check if process exists
            return True
    except (OSError, subprocess.SubprocessError):
        return False

def kill_background_process():
    """Kill the background process safely."""
    pid = get_process_pid()
    
    if pid is None:
        print("❌ No PID file found. No background process to kill.")
        return False
    
    if not is_process_running(pid):
        print(f"❌ Process {pid} is not running.")
        # Clean up stale PID file
        try:
            os.remove(BACKGROUND_PID_FILE)
            print("🧹 Cleaned up stale PID file.")
        except OSError:
            pass
        return False
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
        else:  # Unix-like
            os.kill(pid, signal.SIGTERM)
            # Wait a bit, then force kill if needed
            import time
            time.sleep(2)
            if is_process_running(pid):
                os.kill(pid, signal.SIGKILL)
        
        print(f"✅ Successfully killed background process {pid}")
        
        # Clean up PID file
        try:
            os.remove(BACKGROUND_PID_FILE)
            print("🧹 Cleaned up PID file.")
        except OSError:
            pass
        
        return True
        
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"❌ Failed to kill process {pid}: {e}")
        return False

def status():
    """Show status of background process."""
    pid = get_process_pid()
    
    if pid is None:
        print("📊 Status: No background process found (no PID file)")
        return
    
    if is_process_running(pid):
        print(f"📊 Status: Background process running (PID: {pid})")
        
        # Show log file info if it exists
        if os.path.exists(BACKGROUND_LOG_FILE):
            try:
                stat = os.stat(BACKGROUND_LOG_FILE)
                size_mb = stat.st_size / (1024 * 1024)
                print(f"📝 Log file: {BACKGROUND_LOG_FILE} ({size_mb:.2f} MB)")
            except OSError:
                print(f"📝 Log file: {BACKGROUND_LOG_FILE} (size unknown)")
        else:
            print("📝 Log file: Not found")
            
    else:
        print(f"📊 Status: Process {pid} not running (stale PID file)")
        # Clean up stale PID file
        try:
            os.remove(BACKGROUND_PID_FILE)
            print("🧹 Cleaned up stale PID file.")
        except OSError:
            pass

def tail_log():
    """Show the last few lines of the log file."""
    if not os.path.exists(BACKGROUND_LOG_FILE):
        print(f"❌ Log file {BACKGROUND_LOG_FILE} not found")
        return
    
    try:
        if os.name == 'nt':  # Windows
            # Use PowerShell to tail the file
            subprocess.run([
                "powershell", "-Command", 
                f"Get-Content '{BACKGROUND_LOG_FILE}' -Tail 20"
            ])
        else:  # Unix-like
            subprocess.run(["tail", "-20", BACKGROUND_LOG_FILE])
    except subprocess.CalledProcessError:
        print(f"❌ Failed to tail log file {BACKGROUND_LOG_FILE}")

def follow_log():
    """Follow the log file in real-time."""
    if not os.path.exists(BACKGROUND_LOG_FILE):
        print(f"❌ Log file {BACKGROUND_LOG_FILE} not found")
        return
    
    try:
        if os.name == 'nt':  # Windows
            # Use PowerShell to follow the file
            print(f"📡 Following log file {BACKGROUND_LOG_FILE} (Ctrl+C to stop)...")
            subprocess.run([
                "powershell", "-Command", 
                f"Get-Content '{BACKGROUND_LOG_FILE}' -Wait -Tail 0"
            ])
        else:  # Unix-like
            print(f"📡 Following log file {BACKGROUND_LOG_FILE} (Ctrl+C to stop)...")
            subprocess.run(["tail", "-f", BACKGROUND_LOG_FILE])
    except KeyboardInterrupt:
        print("\n👋 Stopped following log file")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to follow log file {BACKGROUND_LOG_FILE}")

def kill_all_python_processes():
    """Emergency function to kill all Python processes (use with caution!)."""
    print("⚠️  WARNING: This will kill ALL Python processes!")
    response = input("Are you sure? Type 'YES' to confirm: ")
    
    if response != 'YES':
        print("❌ Cancelled.")
        return
    
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], check=True)
            print("✅ Killed all python.exe processes")
        else:  # Unix-like
            subprocess.run(["pkill", "-f", "python"], check=True)
            print("✅ Killed all python processes")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to kill processes: {e}")

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Background Process Manager for Static Experiment")
        print("\nUsage:")
        print("  python background_manager.py status       - Show process status")
        print("  python background_manager.py kill         - Kill background process")
        print("  python background_manager.py tail         - Show last 20 lines of log")
        print("  python background_manager.py follow       - Follow log file in real-time")
        print("  python background_manager.py emergency    - Kill ALL Python processes (DANGEROUS!)")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        status()
    elif command == "kill":
        kill_background_process()
    elif command == "tail":
        tail_log()
    elif command == "follow":
        follow_log()
    elif command == "emergency":
        kill_all_python_processes()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: status, kill, tail, follow, emergency")

if __name__ == "__main__":
    main()
