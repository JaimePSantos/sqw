#!/usr/bin/env python3
"""
Test script to verify streaming memory management for quantum walks.
This tests that memory is properly released between time steps.
"""

import os
import sys
import gc
import psutil
import numpy as np
import time

# Add sqw module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sqw'))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def test_streaming_memory():
    """Test memory usage during streaming computation"""
    from sqw.experiments_expanded_static import running_streaming
    
    # Test parameters - smaller than cluster but large enough to see memory behavior
    N = 1000  # 1000 nodes
    theta = np.pi / 3
    steps = 100  # 100 time steps
    deviation_range = 0.1
    
    print(f"Testing streaming memory management:")
    print(f"N={N}, steps={steps}, theta={theta}, deviation={deviation_range}")
    print(f"Expected matrix size: {N}x{N} complex = {N*N*16/(1024**2):.1f}MB")
    print()
    
    # Memory tracking
    memory_samples = []
    step_memories = []
    
    def memory_tracking_callback(step_idx, state):
        """Callback that tracks memory usage at each step"""
        current_memory = get_memory_usage()
        step_memories.append((step_idx, current_memory))
        
        # Verify state size
        if hasattr(state, 'nbytes'):
            state_size_mb = state.nbytes / (1024**2)
        else:
            # For numpy arrays
            state_size_mb = state.size * state.itemsize / (1024**2)
        
        if step_idx % 20 == 0 or step_idx <= 5:  # Log first few and every 20th step
            print(f"  Step {step_idx:3d}: Memory={current_memory:6.1f}MB, State size={state_size_mb:5.1f}MB, State shape={state.shape}")
        
        # Simulate saving (but don't actually save to disk for this test)
        # This mimics what the cluster script does
        state_copy = state.copy()
        del state_copy
        del state  # Delete the reference
        gc.collect()
        
        # Track memory after cleanup
        memory_after_cleanup = get_memory_usage()
        memory_samples.append(memory_after_cleanup)
    
    print("Starting streaming computation...")
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Run streaming computation
    final_state = running_streaming(
        N, theta, steps,
        initial_nodes=[],
        deviation_range=deviation_range,
        step_callback=memory_tracking_callback
    )
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    print(f"\nStreaming computation completed in {end_time - start_time:.2f}s")
    print(f"Memory: Start={start_memory:.1f}MB, End={end_memory:.1f}MB, Change={end_memory-start_memory:+.1f}MB")
    
    # Analyze memory usage
    if memory_samples:
        max_memory = max(memory_samples)
        min_memory = min(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        
        print(f"Memory during computation: Min={min_memory:.1f}MB, Max={max_memory:.1f}MB, Avg={avg_memory:.1f}MB")
        print(f"Memory variation: {max_memory - min_memory:.1f}MB")
        
        # Check if memory is relatively stable (good sign for streaming)
        memory_stability = (max_memory - min_memory) / avg_memory
        if memory_stability < 0.1:  # Less than 10% variation
            print("✓ Memory usage is stable - streaming is working well")
        elif memory_stability < 0.3:  # Less than 30% variation
            print("⚠ Memory usage has some variation but acceptable")
        else:
            print("✗ Memory usage varies significantly - potential memory leak")
    
    # Clean up
    del final_state
    gc.collect()
    final_memory = get_memory_usage()
    print(f"Final memory after cleanup: {final_memory:.1f}MB")

if __name__ == "__main__":
    test_streaming_memory()
