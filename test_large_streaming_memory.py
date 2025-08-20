#!/usr/bin/env python3
"""
Test script to verify streaming memory management for large quantum walks.
Tests memory behavior at the scale used in the cluster (N=20000).
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

def test_large_streaming_memory():
    """Test memory usage during large streaming computation"""
    from sqw.experiments_expanded_static import running_streaming
    
    # Cluster parameters - same as actual cluster job
    N = 20000  # 20k nodes as in cluster
    theta = np.pi / 3
    steps = 50  # Reduced steps for testing, but same scale
    deviation_range = 0.1
    
    print(f"Testing large-scale streaming memory management:")
    print(f"N={N}, steps={steps}, theta={theta}, deviation={deviation_range}")
    
    # Calculate expected memory usage
    state_size_mb = N * 16 / (1024**2)  # Complex numbers are 16 bytes
    evolution_matrix_size_mb = N * N * 16 / (1024**2)  # Evolution operator
    
    print(f"Expected state size: {state_size_mb:.1f}MB")
    print(f"Expected evolution matrix size: {evolution_matrix_size_mb:.1f}MB")
    print(f"Total expected memory: ~{state_size_mb + evolution_matrix_size_mb:.1f}MB")
    print()
    
    # Memory tracking
    memory_samples = []
    max_memory_seen = 0
    
    def memory_tracking_callback(step_idx, state):
        """Callback that tracks memory usage at each step"""
        nonlocal max_memory_seen
        
        current_memory = get_memory_usage()
        max_memory_seen = max(max_memory_seen, current_memory)
        
        # Calculate actual state size
        state_size_mb = state.size * state.itemsize / (1024**2)
        
        if step_idx % 10 == 0 or step_idx <= 5:  # Log first few and every 10th step
            print(f"  Step {step_idx:3d}: Memory={current_memory:7.1f}MB, State={state_size_mb:5.1f}MB, Shape={state.shape}")
        
        # Simulate the exact same operations as cluster script
        # Save to memory (like pickle.dump would do)
        state_copy = state.copy()
        
        # Delete references to help memory management
        del state_copy
        del state
        gc.collect()
        
        # Track memory after cleanup
        memory_after_cleanup = get_memory_usage()
        memory_samples.append(memory_after_cleanup)
    
    print("Starting large-scale streaming computation...")
    print("This will test the same memory pattern as the cluster...")
    
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        # Run streaming computation
        final_state = running_streaming(
            N, theta, steps,
            initial_nodes=[],
            deviation_range=deviation_range,
            step_callback=memory_tracking_callback
        )
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        print(f"\nLarge streaming computation completed in {end_time - start_time:.2f}s")
        print(f"Memory: Start={start_memory:.1f}MB, End={end_memory:.1f}MB, Change={end_memory-start_memory:+.1f}MB")
        print(f"Peak memory during computation: {max_memory_seen:.1f}MB")
        
        # Analyze memory usage
        if memory_samples:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            
            print(f"Memory during computation: Min={min_memory:.1f}MB, Max={max_memory:.1f}MB, Avg={avg_memory:.1f}MB")
            print(f"Memory variation: {max_memory - min_memory:.1f}MB ({100*(max_memory - min_memory)/avg_memory:.1f}%)")
            
            # Check memory stability
            memory_stability = (max_memory - min_memory) / avg_memory
            if memory_stability < 0.05:  # Less than 5% variation
                print("✓ Memory usage is very stable - excellent streaming")
            elif memory_stability < 0.15:  # Less than 15% variation
                print("✓ Memory usage is stable - streaming is working well")
            elif memory_stability < 0.3:  # Less than 30% variation
                print("⚠ Memory usage has some variation but acceptable")
            else:
                print("✗ Memory usage varies significantly - potential memory issue")
        
        # Clean up
        del final_state
        gc.collect()
        final_memory = get_memory_usage()
        print(f"Final memory after cleanup: {final_memory:.1f}MB")
        
        # Memory efficiency analysis
        theoretical_min = start_memory + state_size_mb  # Just one state
        actual_usage = avg_memory - start_memory
        efficiency = theoretical_min / actual_usage if actual_usage > 0 else 0
        
        print(f"\nMemory efficiency analysis:")
        print(f"Theoretical minimum: {theoretical_min:.1f}MB (start + one state)")
        print(f"Actual average usage: {actual_usage:.1f}MB")
        print(f"Efficiency ratio: {efficiency:.2f} ({'Good' if efficiency > 0.5 else 'Needs improvement'})")
        
    except MemoryError as e:
        print(f"✗ Memory error occurred: {e}")
        print("This indicates the streaming approach needs optimization")
    except Exception as e:
        print(f"✗ Error during computation: {e}")

if __name__ == "__main__":
    print("Testing streaming memory management for cluster-scale computation...")
    print("This will verify that the approach works for N=20000 quantum walks.\n")
    test_large_streaming_memory()
