#!/usr/bin/env python3

"""
Mode Switcher for static_cluster_logged.py

This script helps you easily switch between different execution modes
without manually editing the main script.
"""

import os
import sys
import subprocess

def set_mode(mode):
    """Set the execution mode in static_cluster_logged.py"""
    
    main_script = "static_cluster_logged.py"
    if not os.path.exists(main_script):
        print(f"Error: {main_script} not found in current directory")
        return False
    
    # Define mode configurations
    modes = {
        "full": {
            "CALCULATE_SAMPLES_ONLY": False,
            "SKIP_SAMPLE_COMPUTATION": False,
            "ENABLE_PLOTTING": True,
            "CREATE_TAR_ARCHIVE": True
        },
        "samples": {
            "CALCULATE_SAMPLES_ONLY": True,
            "SKIP_SAMPLE_COMPUTATION": False,
            "ENABLE_PLOTTING": False,
            "CREATE_TAR_ARCHIVE": False
        },
        "analysis": {
            "CALCULATE_SAMPLES_ONLY": False,
            "SKIP_SAMPLE_COMPUTATION": True,
            "ENABLE_PLOTTING": True,
            "CREATE_TAR_ARCHIVE": True
        },
        "quick": {
            "CALCULATE_SAMPLES_ONLY": False,
            "SKIP_SAMPLE_COMPUTATION": True,
            "ENABLE_PLOTTING": True,
            "CREATE_TAR_ARCHIVE": False
        },
        "headless": {
            "CALCULATE_SAMPLES_ONLY": False,
            "SKIP_SAMPLE_COMPUTATION": False,
            "ENABLE_PLOTTING": False,
            "CREATE_TAR_ARCHIVE": True
        }
    }
    
    if mode not in modes:
        print(f"Error: Unknown mode '{mode}'")
        print(f"Available modes: {', '.join(modes.keys())}")
        return False
    
    config = modes[mode]
    
    # Read the current script
    with open(main_script, 'r') as f:
        lines = f.readlines()
    
    # Update the configuration lines
    updated_lines = []
    for line in lines:
        updated = False
        for key, value in config.items():
            if line.strip().startswith(f"{key} = "):
                updated_lines.append(f"{key} = {value}  # Set by mode switcher\n")
                updated = True
                print(f"Updated: {key} = {value}")
                break
        if not updated:
            updated_lines.append(line)
    
    # Write back the updated script
    with open(main_script, 'w') as f:
        f.writelines(updated_lines)
    
    print(f"\n✓ Mode set to '{mode}'")
    return True

def run_with_mode(mode):
    """Set mode and run the script"""
    if set_mode(mode):
        print(f"\nRunning script in '{mode}' mode...")
        try:
            subprocess.run([sys.executable, "static_cluster_logged.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Script execution failed with exit code {e.returncode}")
        except KeyboardInterrupt:
            print("\nScript interrupted by user")

def show_modes():
    """Show available modes and their descriptions"""
    print("Available Execution Modes:")
    print("=" * 50)
    print("full     - Complete pipeline: samples + analysis + plots + archive")
    print("samples  - Only compute and save samples (for cluster)")
    print("analysis - Only analyze existing samples (for local)")
    print("quick    - Quick analysis without archiving")
    print("headless - Full pipeline without plotting (for servers)")
    print()
    print("Usage:")
    print("  python mode_switcher.py <mode>           # Set mode and run")
    print("  python mode_switcher.py set <mode>       # Set mode only")
    print("  python mode_switcher.py show             # Show this help")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_modes()
        return
    
    command = sys.argv[1].lower()
    
    if command == "show" or command == "help":
        show_modes()
    elif command == "set":
        if len(sys.argv) < 3:
            print("Error: Please specify a mode")
            show_modes()
        else:
            mode = sys.argv[2].lower()
            set_mode(mode)
    else:
        # Assume command is a mode name
        mode = command
        run_with_mode(mode)

if __name__ == "__main__":
    main()
