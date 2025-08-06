#!/usr/bin/env python3
"""
Quick setup and test script for crash-safe logging.
Ensures everything is installed and runs a basic test.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check for psutil
    try:
        import psutil
        print("✅ psutil available")
    except ImportError:
        print("⚠️  psutil not found - installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
            print("✅ psutil installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install psutil")
            return False
    
    # Check logging module
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from logging_module import crash_safe_log
        print("✅ Enhanced logging module available")
    except ImportError as e:
        print(f"❌ Enhanced logging module not found: {e}")
        return False
    
    return True

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\n🧪 Running quick test...")
    
    try:
        from logging_module import crash_safe_log
        from logging_module.config import update_config
        
        # Use test log directory
        update_config(logs_base_directory="logs_quick_test")
        
        @crash_safe_log(
            log_file_prefix="quick_test",
            heartbeat_interval=1.0,
            log_system_info=True
        )
        def quick_test():
            print("  Running quick test of enhanced logging...")
            import time
            for i in range(3):
                print(f"  Test step {i+1}/3")
                time.sleep(1)
            return "quick_test_success"
        
        result = quick_test()
        print(f"✅ Quick test passed: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup and test function."""
    print("🚀 CRASH-SAFE LOGGING QUICK SETUP & TEST")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed")
        return False
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Quick test failed")
        return False
    
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("✅ Enhanced crash-safe logging is ready to use")
    print("\n📚 Available test suites:")
    print("  🔬 Comprehensive tests:     python comprehensive_crash_tests.py")
    print("  🖥️  Cluster-specific tests: python cluster_specific_crash_tests.py")
    print("  🎯 All tests:              python run_all_crash_tests.py")
    print("  ⚡ Quick tests:            python run_all_crash_tests.py --quick")
    
    print("\n🔧 Diagnostic tools:")
    print("  📊 Check for crashes:      python -m logging_module.crash_safe_logging --check-crashes")
    print("  🛠️  Generate diagnostics:   python -m logging_module.crash_safe_logging --generate-diagnostics")
    
    print("\n📝 Usage in your experiments:")
    print("""
from logging_module import crash_safe_log

@crash_safe_log(
    log_file_prefix="my_quantum_walk",
    heartbeat_interval=30.0,  # For cluster runs
    log_system_info=True
)
def my_experiment():
    # Your quantum walk code here
    pass
""")
    
    print("🎯 Ready for cluster deployment!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
