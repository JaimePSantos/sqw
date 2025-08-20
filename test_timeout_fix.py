#!/usr/bin/env python3
"""
Test script to verify the timeout fixes in static_cluster_logged_mp.py

This script tests the timeout handling improvements on a small problem size
that's safe to run on your local PC.
"""

import os
import sys

# Set safe test parameters
os.environ['FORCE_N_VALUE'] = '100'  # Very small N for testing
os.environ['FORCE_SAMPLES_COUNT'] = '2'  # Only 2 samples
os.environ['ENABLE_PLOTTING'] = 'False'  # Disable plotting for testing
os.environ['CREATE_TAR_ARCHIVE'] = 'False'  # Disable archiving for testing

# Import and run the main script
if __name__ == "__main__":
    print("=== TESTING TIMEOUT FIXES ===")
    print("Running with safe test parameters:")
    print(f"  N = {os.environ['FORCE_N_VALUE']}")
    print(f"  Samples = {os.environ['FORCE_SAMPLES_COUNT']}")
    print("  Plotting = Disabled")
    print("  Archiving = Disabled")
    print("=" * 40)
    
    # Import and run the updated script
    from static_cluster_logged_mp import run_static_experiment
    
    try:
        result = run_static_experiment()
        print("\n=== TEST COMPLETED SUCCESSFULLY ===")
        print(f"Result: {result}")
    except Exception as e:
        print(f"\n=== TEST FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
