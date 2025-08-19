#!/usr/bin/env python3

"""
Quick test of main script functionality
"""

import sys
import os
sys.path.append('.')

# Test with minimal parameters
os.environ['CALCULATE_SAMPLES_ONLY'] = 'True'  # Only compute samples, no analysis
os.environ['ENABLE_PLOTTING'] = 'False'  # Disable plotting
os.environ['CREATE_TAR_ARCHIVE'] = 'False'  # Disable archiving

# Import and run main function
from static_local_logged_mp import run_static_experiment

print("Testing main script with unified structure...")
print("This will create a small test with 1 sample to verify the structure works")

# Temporarily override parameters to make test smaller
import static_local_logged_mp as main_module
original_N = main_module.N
original_samples = main_module.samples
original_devs = main_module.devs

main_module.N = 10  # Small system
main_module.samples = 1  # Just 1 sample
main_module.devs = [0, 0.01]  # Test both no-noise and noise

try:
    result = run_static_experiment()
    print("✅ Main script ran successfully with unified structure!")
    print(f"Result summary: {result['mode']}, devs: {result['devs']}")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    # Restore original parameters
    main_module.N = original_N
    main_module.samples = original_samples
    main_module.devs = original_devs
