#!/usr/bin/env python3
"""
Small test of the modified multiprocessing static noise experiment
"""
import sys
import os
import multiprocessing as mp
import math

# Configure multiprocessing
mp.set_start_method('spawn', force=True)

# Import the modified experiment
from static_cluster_logged_mp import run_static_experiment

# Override configuration for quick test
import static_cluster_logged_mp as mp_module

# Backup original values
original_devs = mp_module.devs
original_samples = mp_module.samples
original_N = mp_module.N
original_steps = mp_module.steps
original_max_processes = mp_module.MAX_PROCESSES
original_calc_samples_only = mp_module.CALCULATE_SAMPLES_ONLY
original_enable_plotting = mp_module.ENABLE_PLOTTING
original_create_archive = mp_module.CREATE_TAR_ARCHIVE
original_run_in_background = mp_module.RUN_IN_BACKGROUND

try:
    # Set test parameters
    mp_module.devs = [
        0,              # No noise
        (0.1, 0.0),     # Range [0.0, 0.1] - new format
        (0.2, 0.5),     # Range [0.1, 0.2] - new format
    ]
    mp_module.samples = 2  # Quick test
    mp_module.N = 50       # Small system
    mp_module.steps = mp_module.N // 4  # Recalculate steps
    mp_module.MAX_PROCESSES = 2  # Use fewer processes for test
    mp_module.CALCULATE_SAMPLES_ONLY = True  # Only compute samples
    mp_module.ENABLE_PLOTTING = False
    mp_module.CREATE_TAR_ARCHIVE = False
    mp_module.RUN_IN_BACKGROUND = False  # Disable background execution for test
    
    # Fix initial_state_kwargs for the small N
    mp_module.initial_state_kwargs = {"nodes": [mp_module.N//2]}
    
    print("Testing new deviation format with multiprocessing...")
    print(f"Test parameters:")
    print(f"  devs = {mp_module.devs}")
    print(f"  samples = {mp_module.samples}")
    print(f"  N = {mp_module.N}")
    print(f"  steps = {mp_module.steps}")
    print(f"  max_processes = {mp_module.MAX_PROCESSES}")
    print("")
    
    # Run the test
    result = run_static_experiment()
    
    print("\n" + "="*50)
    print("MULTIPROCESSING TEST COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Mode: {result['mode']}")
    print(f"Total time: {result['total_time']:.2f} seconds")
    print(f"Completed samples: {result['completed_samples']}")
    print(f"Master log: {result['master_log_file']}")
    print(f"Process logs: {result['process_log_dir']}")
    
    if 'process_results' in result:
        print(f"\nProcess results:")
        for proc_result in result['process_results']:
            if proc_result['success']:
                print(f"  Dev {proc_result['dev']}: SUCCESS ({proc_result['computed_samples']} samples, {proc_result['total_time']:.2f}s)")
            else:
                print(f"  Dev {proc_result['dev']}: FAILED - {proc_result['error']}")

except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore original values
    mp_module.devs = original_devs
    mp_module.samples = original_samples
    mp_module.N = original_N
    mp_module.steps = original_steps
    mp_module.MAX_PROCESSES = original_max_processes
    mp_module.CALCULATE_SAMPLES_ONLY = original_calc_samples_only
    mp_module.ENABLE_PLOTTING = original_enable_plotting
    mp_module.CREATE_TAR_ARCHIVE = original_create_archive
    mp_module.RUN_IN_BACKGROUND = original_run_in_background
    
    print("\nOriginal configuration restored.")
