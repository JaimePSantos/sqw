#!/usr/bin/env python3

"""
Simplified background launcher for static_cluster_logged.py

This script provides a more robust way to run the main script in background
without relying on os.setsid which can cause issues on some systems.

Usage:
  python safe_background_launcher.py [mode] [options]

Modes:
  full      - Complete pipeline: samples + analysis + plots + archive
  samples   - Only compute and save samples (for cluster)
  analysis  - Only analyze existing samples (for local)  
  quick     - Quick analysis without archiving
  headless  - Full pipeline without plotting (for servers)

Options:
  --no-background  - Run in foreground mode
  --help          - Show this help message

Note: This launcher will use the parameters (N, samples, etc.) from static_cluster_logged.py
"""

import os
import sys
import subprocess
import time

def apply_mode_settings(mode):
    """Apply mode-specific environment variables to control the main script"""
    
    # Define mode configurations
    modes = {
        "full": {
            "CALCULATE_SAMPLES_ONLY": "False",
            "SKIP_SAMPLE_COMPUTATION": "False", 
            "ENABLE_PLOTTING": "True",
            "CREATE_TAR_ARCHIVE": "True",
            "ARCHIVE_SAMPLES": "True",
            "ARCHIVE_PROBDIST": "True"
        },
        "samples": {
            "CALCULATE_SAMPLES_ONLY": "True",
            "SKIP_SAMPLE_COMPUTATION": "False",
            "ENABLE_PLOTTING": "False", 
            "CREATE_TAR_ARCHIVE": "True",
            "ARCHIVE_SAMPLES": "True",
            "ARCHIVE_PROBDIST": "False"
        },
        "analysis": {
            "CALCULATE_SAMPLES_ONLY": "False",
            "SKIP_SAMPLE_COMPUTATION": "True",
            "ENABLE_PLOTTING": "True",
            "CREATE_TAR_ARCHIVE": "True",
            "ARCHIVE_SAMPLES": "False",
            "ARCHIVE_PROBDIST": "True"
        },
        "probdist": {
            "CALCULATE_SAMPLES_ONLY": "False",
            "SKIP_SAMPLE_COMPUTATION": "False",
            "COMPUTE_RAW_SAMPLES": "False",
            "COMPUTE_PROBDIST": "True",
            "COMPUTE_STD_DATA": "False",
            "ENABLE_PLOTTING": "False",
            "CREATE_TAR_ARCHIVE": "True",
            "ARCHIVE_SAMPLES": "False",
            "ARCHIVE_PROBDIST": "True"
        },
        "stddata": {
            "CALCULATE_SAMPLES_ONLY": "False",
            "SKIP_SAMPLE_COMPUTATION": "False",
            "COMPUTE_RAW_SAMPLES": "False",
            "COMPUTE_PROBDIST": "False",
            "COMPUTE_STD_DATA": "True",
            "ENABLE_PLOTTING": "True",
            "CREATE_TAR_ARCHIVE": "False",
            "ARCHIVE_SAMPLES": "False",
            "ARCHIVE_PROBDIST": "False"
        },
        "quick": {
            "CALCULATE_SAMPLES_ONLY": "False", 
            "SKIP_SAMPLE_COMPUTATION": "True",
            "ENABLE_PLOTTING": "True",
            "CREATE_TAR_ARCHIVE": "False",
            "ARCHIVE_SAMPLES": "False",
            "ARCHIVE_PROBDIST": "False"
        },
        "headless": {
            "CALCULATE_SAMPLES_ONLY": "False",
            "SKIP_SAMPLE_COMPUTATION": "False", 
            "ENABLE_PLOTTING": "False",
            "CREATE_TAR_ARCHIVE": "True",
            "ARCHIVE_SAMPLES": "True",
            "ARCHIVE_PROBDIST": "True"
        }
    }
    
    if mode not in modes:
        print(f"Warning: Unknown mode '{mode}', using current script settings")
        return {}
    
    config = modes[mode]
    print(f"Applying mode '{mode}' settings:")
    for key, value in config.items():
        print(f"  {key} = {value}")
    
    return config

def show_help():
    """Show help message"""
    print("Safe Background Launcher for static_cluster_logged.py")
    print("=" * 50)
    print("Usage:")
    print("  python safe_background_launcher.py [mode] [options]")
    print()
    print("Modes:")
    print("  full      - Complete pipeline: samples + analysis + plots + archive (both)")
    print("  samples   - Only compute and save samples (archives samples only)")
    print("  analysis  - Only analyze existing samples (archives probdist only)")
    print("  probdist  - Only compute probability distributions from existing samples")
    print("  stddata   - Only compute standard deviation data with plotting")
    print("  quick     - Quick analysis without archiving")
    print("  headless  - Full pipeline without plotting (archives both)")
    print()
    print("Archive Content by Mode:")
    print("  full/headless - Archives both samples and probability distributions")
    print("  samples       - Archives only raw samples data")
    print("  analysis      - Archives only probability distributions")
    print("  probdist      - Archives only probability distributions")
    print("  stddata/quick - No archiving")
    print()
    print("Computation Control by Mode:")
    print("  full/headless - All computation steps enabled")
    print("  samples       - Only raw sample computation")
    print("  analysis      - All analysis steps (probdist + stddata)")
    print("  probdist      - Only probability distribution computation")
    print("  stddata       - Only standard deviation computation + plotting")
    print()
    print("Options:")
    print("  --no-background  - Run in foreground mode")
    print("  --force         - Kill existing process and start new one")
    print("  --help          - Show this help message")
    print()
    print("Examples:")
    print("  python safe_background_launcher.py samples")
    print("  python safe_background_launcher.py probdist --no-background")
    print("  python safe_background_launcher.py stddata --force")
    print()
    print("Note: This launcher preserves all parameters (N, samples, devs, etc.)")
    print("      from the main static_cluster_logged.py file.")
    print("      Fine-grained computation control can be customized by editing")
    print("      COMPUTE_RAW_SAMPLES, COMPUTE_PROBDIST, and COMPUTE_STD_DATA")
    print("      switches in the main script.")

def get_script_parameters():
    """Read parameters from static_cluster_logged.py"""
    try:
        # Import the main script to get current parameters
        import static_cluster_logged
        # Force reload to get latest values
        import importlib
        importlib.reload(static_cluster_logged)
        
        params = {
            'N': getattr(static_cluster_logged, 'N', 20000),
            'samples': getattr(static_cluster_logged, 'samples', 1),
            'steps': getattr(static_cluster_logged, 'steps', None),
            'devs': getattr(static_cluster_logged, 'devs', []),
            'theta': getattr(static_cluster_logged, 'theta', None)
        }
        
        print(f"Current parameters from static_cluster_logged.py:")
        print(f"  N = {params['N']}")
        print(f"  samples = {params['samples']}")
        print(f"  steps = {params['steps']}")
        print(f"  devs = {params['devs']}")
        print(f"  theta = {params['theta']}")
        
        return params
    except Exception as e:
        print(f"Warning: Could not read parameters from static_cluster_logged.py: {e}")
        return None

def launch_background(mode_config=None, force=False):
    """Launch the main script in background with better compatibility"""
    
    # Get the main script path
    main_script = "static_cluster_logged.py"
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found in current directory")
        return False
    
    # Read current parameters from the script
    params = get_script_parameters()
    if params:
        print(f"\n🔧 Using parameters: N={params['N']}, samples={params['samples']}")
    else:
        print("\n⚠️  Could not read parameters - using script defaults")
    
    # Use current Python executable
    python_exe = sys.executable
    log_file = "static_experiment_background.log"
    pid_file = "static_experiment.pid"
    
    print("Starting background execution with simplified launcher...")
    print(f"Python: {python_exe}")
    print(f"Script: {main_script}")
    print(f"Log: {log_file}")
    
    # Check if already running (unless force is specified)
    if not force and os.path.exists(pid_file):
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
                        print(f"Monitor with: tail -f {log_file}")
                        print(f"Kill with: kill {old_pid}")
                        print("Use bg_manager.py to manage the existing process")
                        print("Or use --force to kill and restart")
                        return False
                else:  # Unix-like
                    os.kill(old_pid, 0)
                    print(f"Background process already running (PID: {old_pid})")
                    print(f"Monitor with: tail -f {log_file}")
                    print(f"Kill with: kill {old_pid}")
                    print("Use bg_manager.py to manage the existing process")
                    print("Or use --force to kill and restart")
                    return False
            except OSError:
                # Process doesn't exist, clean up stale PID file
                print(f"Cleaning up stale PID file (process {old_pid} not running)")
                os.remove(pid_file)
        except (ValueError, IOError):
            # Invalid PID file, remove it
            print("Cleaning up invalid PID file")
            try:
                os.remove(pid_file)
            except:
                pass
    elif force and os.path.exists(pid_file):
        # Force mode: kill existing process
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            
            print(f"Force mode: killing existing process (PID: {old_pid})")
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], 
                                 capture_output=True, text=True)
                else:  # Unix-like
                    os.kill(old_pid, 15)  # SIGTERM first
                    time.sleep(1)
                    try:
                        os.kill(old_pid, 0)  # Check if still alive
                        os.kill(old_pid, 9)  # SIGKILL if stubborn
                    except OSError:
                        pass  # Process is dead
                
                print("Existing process killed")
                os.remove(pid_file)
            except (OSError, subprocess.SubprocessError) as e:
                print(f"Warning: Could not kill process {old_pid}: {e}")
                # Continue anyway, maybe it's dead
                try:
                    os.remove(pid_file)
                except:
                    pass
        except (ValueError, IOError):
            # Invalid PID file, just remove it
            try:
                os.remove(pid_file)
            except:
                pass
    
    # Prepare environment to prevent recursion
    env = os.environ.copy()
    env['IS_BACKGROUND_PROCESS'] = '1'
    env['RUN_IN_BACKGROUND'] = 'False'  # Disable the script's own background logic
    
    # Apply mode-specific settings if provided
    if mode_config:
        env.update(mode_config)
    
    # Force current parameters from script (override any cached/existing data behavior)
    if params:
        env['FORCE_SAMPLES_COUNT'] = str(params['samples'])
        env['FORCE_N_VALUE'] = str(params['N'])
        print(f"🔒 Forcing parameters: samples={params['samples']}, N={params['N']}")
    
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
    """Main function with mode support"""
    
    # Parse command line arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    mode = None
    run_foreground = False
    force_launch = False
    
    # Parse arguments
    for arg in args:
        if arg == "--help" or arg == "-h":
            show_help()
            return
        elif arg == "--no-background":
            run_foreground = True
        elif arg == "--force":
            force_launch = True
        elif arg in ["full", "samples", "analysis", "probdist", "stddata", "quick", "headless"]:
            mode = arg
        else:
            print(f"Unknown argument: {arg}")
            show_help()
            return
    
    # Apply mode configuration
    mode_config = None
    if mode:
        mode_config = apply_mode_settings(mode)
        print()
    
    if run_foreground:
        print("Running in foreground mode...")
        # Set environment to disable background execution in main script
        os.environ['RUN_IN_BACKGROUND'] = 'False'
        
        # Apply mode settings to environment
        if mode_config:
            os.environ.update(mode_config)
        
        # Run main script directly
        try:
            import static_cluster_logged
            static_cluster_logged.run_static_experiment()
        except Exception as e:
            print(f"Error running script: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # Background execution
    success = launch_background(mode_config, force_launch)
    if not success:
        print("\\nBackground process launch failed or process already running.")
        print("Check the messages above for details.")
        
        # Only fall back to foreground if specifically requested, not on existing process
        fallback_choice = input("Run in foreground instead? (y/N): ").lower().strip()
        if fallback_choice == 'y' or fallback_choice == 'yes':
            print("Running in foreground mode...")
            os.environ['RUN_IN_BACKGROUND'] = 'False'
            
            # Apply mode settings to environment
            if mode_config:
                os.environ.update(mode_config)
            
            try:
                import static_cluster_logged
                static_cluster_logged.run_static_experiment()
            except Exception as e:
                print(f"Error running script: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Exiting. Use bg_manager.py to manage existing processes.")

if __name__ == "__main__":
    main()
