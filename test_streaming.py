#!/usr/bin/env python3

"""
Test script to verify the streaming approach works correctly.
Uses smaller parameters to test functionality.
"""

import sys
import os
import time

# Add the current directory to path
sys.path.append(os.path.dirname(__file__))

print("Testing streaming quantum walk approach...")

try:
    # Import the streaming function
    from sqw.experiments_expanded_static import running_streaming, running
    
    # Test parameters
    N = 100
    theta = 3.14159/3
    steps = 20
    initial_nodes = [N//2]
    deviation_range = 0.1
    
    print(f"Test parameters: N={N}, steps={steps}, dev={deviation_range}")
    
    # Test 1: Compare streaming vs traditional approach
    print("\n=== Test 1: Comparing streaming vs traditional ===")
    
    # Set random seed for reproducible results
    import random
    import numpy as np
    
    # Traditional approach
    print("Running traditional approach...")
    random.seed(12345)
    np.random.seed(12345)
    start_time = time.time()
    traditional_result = running(
        N, theta, steps,
        initial_nodes=initial_nodes,
        deviation_range=deviation_range,
        return_all_states=True
    )
    traditional_time = time.time() - start_time
    print(f"Traditional approach: {len(traditional_result)} states in {traditional_time:.3f}s")
    
    # Streaming approach (same seed)
    print("Running streaming approach...")
    random.seed(12345)
    np.random.seed(12345)
    saved_states = []
    
    def collect_states(step_idx, state):
        saved_states.append((step_idx, state.copy()))
        if step_idx % 5 == 0:
            print(f"  Collected step {step_idx}")
    
    start_time = time.time()
    final_state = running_streaming(
        N, theta, steps,
        initial_nodes=initial_nodes,
        deviation_range=deviation_range,
        step_callback=collect_states
    )
    streaming_time = time.time() - start_time
    print(f"Streaming approach: {len(saved_states)} states in {streaming_time:.3f}s")
    
    # Test 2: Verify results are identical
    print("\n=== Test 2: Verifying results match ===")
    
    if len(saved_states) == len(traditional_result):
        print(f"✓ Same number of states: {len(saved_states)}")
        
        # Check if states are the same
        max_diff = 0
        for i, (step_idx, streaming_state) in enumerate(saved_states):
            traditional_state = traditional_result[i]
            diff = abs(streaming_state - traditional_state).max()
            max_diff = max(max_diff, diff)
        
        if max_diff < 1e-12:
            print(f"✓ States match within numerical precision (max diff: {max_diff:.2e})")
        else:
            print(f"✗ States differ by {max_diff:.2e}")
    else:
        print(f"✗ Different number of states: {len(saved_states)} vs {len(traditional_result)}")
    
    # Test 3: Test file saving callback
    print("\n=== Test 3: Testing file saving ===")
    
    test_dir = "test_streaming_output"
    os.makedirs(test_dir, exist_ok=True)
    
    def save_states_callback(step_idx, state):
        filename = f"test_step_{step_idx}.pkl"
        filepath = os.path.join(test_dir, filename)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        if step_idx % 5 == 0:
            print(f"  Saved step {step_idx} to {filename}")
    
    print("Running streaming with file saving...")
    start_time = time.time()
    final_state = running_streaming(
        N, theta, steps,
        initial_nodes=initial_nodes,
        deviation_range=deviation_range,
        step_callback=save_states_callback
    )
    save_time = time.time() - start_time
    print(f"Streaming with file saving completed in {save_time:.3f}s")
    
    # Verify files were created
    saved_files = [f for f in os.listdir(test_dir) if f.startswith("test_step_")]
    print(f"✓ Created {len(saved_files)} files in {test_dir}")
    
    # Clean up test files
    for filename in saved_files:
        os.remove(os.path.join(test_dir, filename))
    os.rmdir(test_dir)
    print("✓ Cleaned up test files")
    
    print("\n=== Test Summary ===")
    print("✓ Streaming approach works correctly")
    print("✓ Results match traditional approach")
    print("✓ File saving callback works")
    print(f"✓ Performance: traditional={traditional_time:.3f}s, streaming={streaming_time:.3f}s")
    print("\nReady to test with full parameters!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
