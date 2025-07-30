"""
Example usage of the crash_safe_logging decorator module.

This file demonstrates different ways to use the crash-safe logging system.
"""

import time
import logging
from crash_safe_logging import crash_safe_log, setup_logging


# Example 1: Basic usage with default settings
@crash_safe_log()
def simple_function():
    """Simple function with default logging settings."""
    print("Running simple function...")
    time.sleep(2)
    return "Simple function completed"


# Example 2: Custom settings
@crash_safe_log(
    log_file_prefix="custom_task",
    heartbeat_interval=5.0,
    log_level=logging.INFO
)
def custom_function():
    """Function with custom logging settings."""
    print("Running custom function...")
    for i in range(10):
        print(f"Processing item {i+1}/10")
        time.sleep(1)
    return "Custom function completed"


# Example 3: Function that might crash
@crash_safe_log(log_file_prefix="risky_task")
def risky_function(should_crash=False):
    """Function that might crash to test error handling."""
    print("Running risky function...")
    time.sleep(2)
    
    if should_crash:
        raise ValueError("Intentional crash for testing!")
    
    return "Risky function completed successfully"


# Example 4: Manual logging setup (alternative to decorator)
def manual_logging_example():
    """Example of manual logging setup without decorator."""
    logger, crash_logger = setup_logging(log_file_prefix="manual_setup")
    
    try:
        logger.info("Starting manual logging example")
        
        # Log system info
        crash_logger.log_system_info()
        
        logger.info("Doing some work...")
        time.sleep(3)
        
        logger.info("Work completed successfully")
        return "Manual logging example completed"
        
    except Exception as e:
        logger.error(f"Error in manual example: {e}")
        raise
    finally:
        # Always cleanup
        crash_logger.cleanup()


if __name__ == "__main__":
    print("=== Testing crash-safe logging decorator ===\n")
    
    # Test 1: Simple function
    print("1. Testing simple function...")
    result1 = simple_function()
    print(f"Result: {result1}\n")
    
    # Test 2: Custom function
    print("2. Testing custom function...")
    result2 = custom_function()
    print(f"Result: {result2}\n")
    
    # Test 3: Function without crash
    print("3. Testing risky function (no crash)...")
    result3 = risky_function(should_crash=False)
    print(f"Result: {result3}\n")
    
    # Test 4: Manual logging
    print("4. Testing manual logging setup...")
    result4 = manual_logging_example()
    print(f"Result: {result4}\n")
    
    # Test 5: Function with crash (uncomment to test)
    # print("5. Testing risky function (with crash)...")
    # try:
    #     result5 = risky_function(should_crash=True)
    # except ValueError as e:
    #     print(f"Caught expected error: {e}")
    
    print("All tests completed! Check the generated log files for details.")
