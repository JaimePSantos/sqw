"""
Test script to verify the organized logging folder structure.
"""

from crash_safe_logging import crash_safe_log, print_log_summary, get_latest_log_file
import time

@crash_safe_log(log_file_prefix="test_organized")
def test_organized_logging():
    """Test function to verify organized logging structure."""
    print("Testing organized logging structure...")
    time.sleep(2)
    print("Test completed!")
    return "organized_test_success"

if __name__ == "__main__":
    print("Testing organized logging folder structure...\n")
    
    # Run test with organized logging
    result = test_organized_logging()
    print(f"Test result: {result}\n")
    
    # Show log summary
    print_log_summary()
    
    # Show latest log file
    latest = get_latest_log_file()
    if latest:
        print(f"\nLatest log file: {latest}")
    
    print("\nCheck the 'logs' directory to see the organized structure!")
    print("Structure should be: logs/YYYY-MM-DD/test_organized_HH-MM-SS.log")
