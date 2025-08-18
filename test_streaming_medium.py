#!/usr/bin/env python3

"""
Test script with medium-sized parameters to verify memory efficiency.
"""

import sys
import os
import time
import psutil

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

print("Testing streaming approach with medium-sized parameters...")

try:
    # Import the streaming function
    from sqw.experiments_expanded_static import running_streaming
    
    # Medium test parameters
    N = 1000
    theta = 3.14159/3
    steps = 200  # Reasonable size
    initial_nodes = [N//2]
    deviation_range = 0.1
    
    print(f"Medium test parameters: N={N}, steps={steps}, dev={deviation_range}")
    
    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    saved_count = [0]  # Use list to make it mutable
    max_memory = [initial_memory]
    
    def monitor_and_save(step_idx, state):
        saved_count[0] += 1
        
        # Monitor memory
        current_memory = process.memory_info().rss / (1024 * 1024)
        max_memory[0] = max(max_memory[0], current_memory)
        
        # Progress reporting
        if step_idx % 50 == 0 or step_idx == steps:
            print(f"  Step {step_idx}: {current_memory:.1f} MB RAM")
    
    print("Running streaming quantum walk...")
    start_time = time.time()
    
    final_state = running_streaming(
        N, theta, steps,
        initial_nodes=initial_nodes,
        deviation_range=deviation_range,
        step_callback=monitor_and_save
    )
    
    duration = time.time() - start_time
    final_memory = process.memory_info().rss / (1024 * 1024)
    
    print(f"\n=== Results ===")
    print(f"✓ Completed in {duration:.2f} seconds")
    print(f"✓ Processed {saved_count[0]} states")
    print(f"✓ Initial memory: {initial_memory:.1f} MB")
    print(f"✓ Peak memory: {max_memory[0]:.1f} MB")
    print(f"✓ Final memory: {final_memory:.1f} MB")
    print(f"✓ Memory increase: {max_memory[0] - initial_memory:.1f} MB")
    
    # Estimate what traditional approach would need
    traditional_memory_estimate = (steps * N * 16) / (1024 * 1024)  # 16 bytes per complex number
    print(f"✓ Traditional approach would need ~{traditional_memory_estimate:.1f} MB for states alone")
    
    if max_memory[0] - initial_memory < traditional_memory_estimate / 10:
        print("✓ Streaming approach uses much less memory than traditional!")
    else:
        print("⚠ Memory usage higher than expected")
    
    print("\nReady for full-scale testing!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
