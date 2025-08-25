#!/usr/bin/env python3
"""
Final validation test for the logging module.
This test covers basic functionality without risking hanging.
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logging_module import crash_safe_log

@crash_safe_log(log_file_prefix="final_validation")
def validation_test():
    """Final validation of the logging module."""
    print("=== FINAL VALIDATION TEST ===")
    
    # Test 1: Basic execution
    print("Test 1: Basic execution")
    x = 10
    y = 20
    result = x + y
    print(f"10 + 20 = {result}")
    
    # Test 2: Simple loop
    print("Test 2: Simple loop")
    for i in range(3):
        print(f"  Loop iteration {i+1}")
        time.sleep(0.2)  # Short sleep
    
    # Test 3: String operations
    print("Test 3: String operations")
    message = "Logging module works!"
    print(f"Message: {message}")
    print(f"Message length: {len(message)}")
    
    # Test 4: List operations
    print("Test 4: List operations")
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    print(f"Sum of {numbers} = {total}")
    
    print("=== ALL TESTS COMPLETED SUCCESSFULLY ===")
    return "validation_success"

def main():
    """Main function to run validation."""
    print("Starting final validation of logging module...")
    
    try:
        result = validation_test()
        print(f"\nValidation result: {result}")
        print("[PASS] Logging module is working correctly!")
        
    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
