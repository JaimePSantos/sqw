#!/usr/bin/env python3

"""
Quick test to check the multiprocessing configuration and parameters.
"""

import multiprocessing as mp

# Import the parameters from the main script
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Read the configuration from the main script
print("Testing multiprocessing configuration...")

try:
    # Check CPU count and memory
    cpu_count = mp.cpu_count()
    print(f"Available CPUs: {cpu_count}")
    
    # Read current parameters from the main script
    with open('static_cluster_logged_mp.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Extract N and steps
    for line in content.split('\n'):
        if line.strip().startswith('N = '):
            n_line = line.strip()
            print(f"System size: {n_line}")
        elif line.strip().startswith('steps = '):
            steps_line = line.strip()
            print(f"Time steps: {steps_line}")
        elif line.strip().startswith('samples = '):
            samples_line = line.strip()
            print(f"Samples: {samples_line}")
        elif line.strip().startswith('MAX_PROCESSES = '):
            max_proc_line = line.strip()
            print(f"Max processes: {max_proc_line}")
            
    # Calculate estimated memory usage
    N = 20000
    steps = min(N//4, 1000)
    samples = 5
    devs = [0, 0.1, 0.5, 1, 10]
    max_processes = min(len(devs), max(1, cpu_count // 2))
    
    print(f"\nCalculated values:")
    print(f"N = {N}")
    print(f"steps = {steps}")
    print(f"samples = {samples}")
    print(f"max_processes = {max_processes}")
    print(f"deviation values = {devs}")
    
    # Estimate memory usage per process
    # Each state is approximately N complex numbers = N * 16 bytes
    # Each sample stores 'steps' states = steps * N * 16 bytes
    state_size_mb = (N * 16) / (1024 * 1024)
    sample_size_mb = (steps * N * 16) / (1024 * 1024)
    
    print(f"\nMemory estimation:")
    print(f"Size per state: {state_size_mb:.1f} MB")
    print(f"Size per sample (all states): {sample_size_mb:.1f} MB")
    print(f"Max memory per process: ~{sample_size_mb * 2:.1f} MB (with overhead)")
    print(f"Total estimated memory: ~{sample_size_mb * 2 * max_processes:.1f} MB")
    
    if sample_size_mb > 1000:
        print("⚠️  WARNING: Large memory usage detected!")
        print("   Consider reducing N or steps to prevent memory issues.")
    else:
        print("✓ Memory usage looks reasonable.")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
