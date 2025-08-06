#!/usr/bin/env python3
"""
Installation script for enhanced crash-safe logging.
Ensures all dependencies are installed and provides setup verification.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("=== Checking Python Version ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 7):
        print("[FAIL] Error: Python 3.7 or higher is required")
        return False
    else:
        print("[PASS] Python version is compatible")
        return True

def install_dependencies():
    """Install required Python packages."""
    print("\n=== Installing Dependencies ===")
    
    packages = [
        "psutil",  # For system monitoring
    ]
    
    success = True
    for package in packages:
        print(f"\nInstalling {package}...")
        if not run_command(f"{sys.executable} -m pip install {package}", f"Install {package}"):
            print(f"[FAIL] Failed to install {package}")
            success = False
        else:
            print(f"[PASS] Successfully installed {package}")
    
    return success

def verify_installation():
    """Verify that all components are working."""
    print("\n=== Verifying Installation ===")
    
    try:
        # Test basic imports
        print("Testing imports...")
        import logging
        import multiprocessing
        import threading
        import signal
        print("[PASS] Basic imports successful")
        
        # Test psutil import
        try:
            import psutil
            print("[PASS] psutil import successful")
            
            # Test basic psutil functionality
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"[PASS] psutil functionality test: Memory usage = {memory_info.rss / 1024 / 1024:.1f} MB")
        except ImportError:
            print("[WARN] psutil not available - some monitoring features will be limited")
        except Exception as e:
            print(f"[WARN] psutil test failed: {e}")
        
        # Test logging module
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        try:
            from logging_module import crash_safe_log
            print("[PASS] Enhanced logging module import successful")
        except ImportError as e:
            print(f"[FAIL] Failed to import logging module: {e}")
            return False
        
        print("[PASS] All verifications passed!")
        return True
        
    except Exception as e:
        print(f"[FAIL] Verification failed: {e}")
        return False

def create_test_script():
    """Create a simple test script."""
    print("\n=== Creating Test Script ===")
    
    test_script = """#!/usr/bin/env python3
'''Simple test for enhanced crash-safe logging.'''

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from logging_module import crash_safe_log

@crash_safe_log(
    log_file_prefix="installation_test",
    heartbeat_interval=2.0,
    log_system_info=True
)
def test_logging():
    '''Test the enhanced logging system.'''
    print("Testing enhanced crash-safe logging...")
    
    for i in range(5):
        print(f"Test iteration {i+1}/5")
        time.sleep(1)
    
    print("Test completed successfully!")
    return "success"

if __name__ == "__main__":
    result = test_logging()
    print(f"Test result: {result}")
"""
    
    try:
        with open("test_logging_installation.py", "w") as f:
            f.write(test_script)
        os.chmod("test_logging_installation.py", 0o755)
        print("[PASS] Test script created: test_logging_installation.py")
        print("   Run with: python test_logging_installation.py")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to create test script: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions."""
    print("\n" + "="*60)
    print("ENHANCED CRASH-SAFE LOGGING - INSTALLATION COMPLETE")
    print("="*60)
    
    print("\n📚 USAGE EXAMPLES:")
    print("\n1. Basic usage:")
    print("""
from logging_module import crash_safe_log

@crash_safe_log()
def my_experiment():
    # Your code here
    pass
""")
    
    print("\n2. Cluster-optimized usage:")
    print("""
@crash_safe_log(
    log_file_prefix="quantum_walk",
    heartbeat_interval=30.0,  # Longer intervals for cluster
    log_system_info=True
)
def run_cluster_experiment():
    # Your long-running experiment
    pass
""")
    
    print("\n[TOOL] DIAGNOSTIC TOOLS:")
    print("  Check for crashes:     python -m logging_module.crash_safe_logging --check-crashes")
    print("  Generate diagnostics:  python -m logging_module.crash_safe_logging --generate-diagnostics")
    print("  Test installation:     python test_logging_installation.py")
    
    print("\n📋 CLUSTER TROUBLESHOOTING:")
    print("  See: logging_module/CLUSTER_TROUBLESHOOTING.md")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Run the test script to verify everything works")
    print("  2. Review the troubleshooting guide for cluster-specific issues")
    print("  3. Add the @crash_safe_log decorator to your experiment functions")
    print("  4. Monitor logs in the 'logs/' directory")

def main():
    """Main installation function."""
    print("Enhanced Crash-Safe Logging Installation")
    print("=======================================")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("[FAIL] Dependency installation failed")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("[FAIL] Installation verification failed")
        sys.exit(1)
    
    # Create test script
    create_test_script()
    
    # Print usage instructions
    print_usage_instructions()
    
    print("\n🎉 Installation completed successfully!")

if __name__ == "__main__":
    main()
