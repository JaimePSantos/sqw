#!/usr/bin/env python3
"""
Test script for enhanced cluster-aware crash-safe logging.
This script simulates different termination scenarios to test logging capabilities.
"""

import time
import sys
import os

# Add the parent directory to the path so we can import the logging module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from logging_module import crash_safe_log


@crash_safe_log(
    log_file_prefix="cluster_test",
    heartbeat_interval=5.0,  # Frequent heartbeats for testing
    log_system_info=True
)
def test_normal_execution():
    """Test normal execution with proper shutdown."""
    print("Starting normal execution test...")
    
    for i in range(10):
        print(f"Working on iteration {i+1}/10")
        time.sleep(2)  # Simulate work
    
    print("Normal execution completed successfully!")
    return "success"


@crash_safe_log(
    log_file_prefix="memory_test",
    heartbeat_interval=3.0,
    log_system_info=True
)
def test_memory_usage():
    """Test with gradually increasing memory usage."""
    print("Starting memory usage test...")
    
    # Gradually allocate memory to test monitoring
    data = []
    for i in range(50):
        # Allocate ~10MB per iteration
        chunk = [0] * (10 * 1024 * 1024 // 8)  # 10MB of integers
        data.append(chunk)
        print(f"Allocated {(i+1) * 10} MB")
        time.sleep(2)
    
    print("Memory test completed!")
    return "success"


@crash_safe_log(
    log_file_prefix="interrupt_test", 
    heartbeat_interval=2.0,
    log_system_info=True
)
def test_keyboard_interrupt():
    """Test keyboard interrupt handling."""
    print("Starting interrupt test...")
    print("Press Ctrl+C within the next 30 seconds to test interrupt handling...")
    
    for i in range(15):
        print(f"Waiting for interrupt... {i+1}/15")
        time.sleep(2)
    
    print("No interrupt received - test completed normally")
    return "success"


def main():
    """Main test function."""
    if len(sys.argv) < 2:
        print("Usage: python test_enhanced_logging.py <test_type>")
        print("Test types:")
        print("  normal     - Test normal execution")
        print("  memory     - Test memory usage monitoring")
        print("  interrupt  - Test keyboard interrupt handling")
        print("  crash      - Check for previous crashes")
        print("  diagnostics - Generate cluster diagnostics")
        return
    
    test_type = sys.argv[1].lower()
    
    if test_type == "normal":
        result = test_normal_execution()
        print(f"Test result: {result}")
        
    elif test_type == "memory":
        try:
            result = test_memory_usage()
            print(f"Test result: {result}")
        except KeyboardInterrupt:
            print("Memory test interrupted by user")
        except MemoryError:
            print("Memory test hit memory limit")
            
    elif test_type == "interrupt":
        try:
            result = test_keyboard_interrupt()
            print(f"Test result: {result}")
        except KeyboardInterrupt:
            print("Interrupt test - caught keyboard interrupt successfully!")
            
    elif test_type == "crash":
        from logging_module.crash_safe_logging import check_for_crashed_processes
        check_for_crashed_processes()
        
    elif test_type == "diagnostics":
        from logging_module.crash_safe_logging import generate_cluster_diagnostic_script
        generate_cluster_diagnostic_script()
        
    else:
        print(f"Unknown test type: {test_type}")
        print("Use 'normal', 'memory', 'interrupt', 'crash', or 'diagnostics'")


if __name__ == "__main__":
    main()
