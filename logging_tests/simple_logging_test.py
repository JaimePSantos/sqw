"""
Simple test for the logging module to verify it works without hanging.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_module import crash_safe_log
import time

@crash_safe_log(log_file_prefix="simple_test", heartbeat_interval=2.0)
def quick_test():
    """Quick test function that should complete in a few seconds."""
    print("Starting quick test...")
    time.sleep(3)  # Short sleep
    print("Quick test completed!")
    return "test_success"

if __name__ == "__main__":
    print("Running simple test of logging module...")
    result = quick_test()
    print(f"Test result: {result}")
    print("Test completed successfully!")
