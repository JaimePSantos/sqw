#!/usr/bin/env python3
"""
Test the sparse matrix implementation for large quantum walks.
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

def test_sparse_implementation():
    """Test the sparse matrix implementation"""
    
    # First test memory estimation
    print("Testing sparse matrix memory estimation:")
    from sqw.experiments_sparse import estimate_sparse_memory_usage
    
    for N in [1000, 5000, 10000, 20000]:
        mem = estimate_sparse_memory_usage(N)
        print(f"N={N:5d}: Sparse matrix={mem['sparse_matrix_mb']:6.1f}MB, "
              f"State={mem['state_mb']:5.1f}MB, Total={mem['total_mb']:7.1f}MB")
    
    print("\nTesting sparse implementation with N=20000...")
    
    from sqw.experiments_sparse import running_streaming_sparse
    
    # Test parameters
    N = 20000
    theta = np.pi / 3
    steps = 10  # Small number of steps for testing
    deviation_range = 0.1
    
    print(f"Parameters: N={N}, steps={steps}, theta={theta}, deviation={deviation_range}")
    
    # Memory tracking
    memory_samples = []
    step_count = 0
    
    def memory_tracking_callback(step_idx, state):
        """Callback that tracks memory usage at each step"""
        nonlocal step_count
        step_count += 1
        
        current_memory = get_memory_usage()
        memory_samples.append(current_memory)
        
        # Calculate state size
        state_size_mb = state.size * state.itemsize / (1024**2)
        
        print(f"  Step {step_idx:3d}: Memory={current_memory:7.1f}MB, State={state_size_mb:5.1f}MB")
        
        # Clean up reference
        del state
        gc.collect()
    
    print("\nStarting sparse streaming computation...")
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        # Run sparse streaming computation
        final_state = running_streaming_sparse(
            N, theta, steps,
            initial_nodes=[],
            deviation_range=deviation_range,
            step_callback=memory_tracking_callback
        )
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        print(f"\nSparse streaming computation completed in {end_time - start_time:.2f}s")
        print(f"Memory: Start={start_memory:.1f}MB, End={end_memory:.1f}MB, Change={end_memory-start_memory:+.1f}MB")
        
        if memory_samples:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            
            print(f"Memory during computation: Min={min_memory:.1f}MB, Max={max_memory:.1f}MB, Avg={avg_memory:.1f}MB")
            print(f"Memory variation: {max_memory - min_memory:.1f}MB ({100*(max_memory - min_memory)/avg_memory:.1f}%)")
            
            # Success!
            print("✓ Sparse matrix implementation works for N=20000!")
        
        # Clean up
        del final_state
        gc.collect()
        final_memory = get_memory_usage()
        print(f"Final memory after cleanup: {final_memory:.1f}MB")
        
        return True
        
    except MemoryError as e:
        print(f"✗ Memory error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sparse_implementation()
    if success:
        print("\n✓ Sparse implementation is ready for cluster deployment!")
    else:
        print("\n✗ Sparse implementation needs further optimization.")
