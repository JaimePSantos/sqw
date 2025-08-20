#!/usr/bin/env python3
"""
Specific test for N=20k to verify we're only storing one matrix at a time,
not 5x20kx20k matrices.
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

def test_n20k_memory_safety():
    """Test N=20k to ensure we're not storing multiple large matrices"""
    print("=== N=20k MEMORY SAFETY TEST ===")
    
    N = 20000
    print(f"Testing N={N} for memory safety patterns")
    
    # Import required modules
    from sqw.experiments_sparse import running_streaming_sparse
    
    # Memory checkpoints
    checkpoints = []
    states_seen = []
    max_memory = 0
    
    def memory_monitor_callback(step_idx, state):
        """Monitor memory and state sizes during streaming"""
        nonlocal max_memory
        
        current_memory = get_memory_usage()
        max_memory = max(max_memory, current_memory)
        
        # Record detailed info for first few and last few steps
        if step_idx <= 3 or step_idx >= 17:
            state_size_mb = state.size * state.itemsize / (1024**2)
            checkpoints.append({
                'step': step_idx,
                'memory_mb': current_memory,
                'state_size_mb': state_size_mb,
                'state_shape': state.shape,
                'state_dtype': state.dtype
            })
            
            print(f"    Step {step_idx:2d}: Memory={current_memory:6.1f}MB, "
                  f"State={state_size_mb:.3f}MB, Shape={state.shape}")
        
        # Track state reference count (should always be minimal)
        states_seen.append(step_idx)
        
        # Verify this is the only large object in memory
        if step_idx == 0:
            # First state - establish baseline
            baseline_memory = current_memory
            print(f"      Baseline memory with first state: {baseline_memory:.1f}MB")
        
        # Clean up reference immediately
        del state
        gc.collect()
    
    print(f"Starting N={N} streaming test with 20 steps...")
    print("This will verify we NEVER store multiple 20k×20k matrices simultaneously")
    
    start_memory = get_memory_usage()
    start_time = time.time()
    
    try:
        # Run the streaming computation
        final_state = running_streaming_sparse(
            N, np.pi/3, 20,  # 20 steps for testing
            initial_nodes=[],
            deviation_range=0.1,
            step_callback=memory_monitor_callback
        )
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        print(f"\nN={N} streaming test completed in {end_time - start_time:.1f}s")
        print(f"Memory: Start={start_memory:.1f}MB, End={end_memory:.1f}MB, Change={end_memory-start_memory:+.1f}MB")
        print(f"Peak memory during computation: {max_memory:.1f}MB")
        
        # Analyze the results
        print(f"\n--- MEMORY SAFETY ANALYSIS ---")
        
        # Check if we're storing reasonable amounts
        expected_state_size = N * 16 / (1024**2)  # One complex state vector
        expected_sparse_matrices = 3 * 0.6  # ~0.6MB per sparse matrix, 3 matrices
        expected_total = expected_state_size + expected_sparse_matrices
        
        actual_peak_increase = max_memory - start_memory
        
        print(f"Expected memory for streaming N={N}:")
        print(f"  State vector: {expected_state_size:.1f}MB")
        print(f"  Sparse matrices (3x): {expected_sparse_matrices:.1f}MB")
        print(f"  Expected total: {expected_total:.1f}MB")
        print(f"Actual peak memory increase: {actual_peak_increase:.1f}MB")
        
        # Safety checks
        safety_ratio = actual_peak_increase / expected_total
        print(f"Memory efficiency ratio: {safety_ratio:.2f} (1.0 = perfect, <2.0 = good)")
        
        if actual_peak_increase < 10:  # Less than 10MB total
            print("✓ EXCELLENT: Memory usage is minimal")
        elif actual_peak_increase < 50:  # Less than 50MB total
            print("✓ GOOD: Memory usage is reasonable")
        elif actual_peak_increase < 200:  # Less than 200MB total
            print("⚠ ACCEPTABLE: Memory usage is moderate")
        else:
            print("✗ WARNING: Memory usage is high")
        
        # Check for 5x20kx20k pattern (would be ~120GB)
        danger_threshold = 1000  # 1GB - way more than we should ever use
        if actual_peak_increase > danger_threshold:
            print(f"✗ DANGER: Memory usage ({actual_peak_increase:.1f}MB) suggests multiple large matrices!")
        else:
            print(f"✓ SAFE: No evidence of storing multiple large matrices")
        
        # Verify state consistency
        if checkpoints:
            state_sizes = [cp['state_size_mb'] for cp in checkpoints]
            if len(set(f"{s:.3f}" for s in state_sizes)) == 1:
                print("✓ CONSISTENT: All states have identical size")
            else:
                print("⚠ VARIATION: State sizes vary (might be normal)")
        
        # Final state check
        final_state_size = final_state.size * final_state.itemsize / (1024**2)
        print(f"Final state size: {final_state_size:.3f}MB")
        
        del final_state
        gc.collect()
        
        cleanup_memory = get_memory_usage()
        print(f"Memory after cleanup: {cleanup_memory:.1f}MB")
        
        # Summary
        print(f"\n--- SUMMARY FOR N={N} ---")
        print(f"Steps processed: {len(states_seen)}")
        print(f"Peak memory: {max_memory:.1f}MB")
        print(f"Memory per step: {actual_peak_increase:.1f}MB")
        print(f"Time per step: {(end_time - start_time) / 20:.2f}s")
        
        if actual_peak_increase < 100:
            print("🎉 SUCCESS: Memory usage confirms single-matrix streaming pattern!")
            print("   No evidence of storing 5×20k×20k matrices (which would need ~120GB)")
            return True
        else:
            print("⚠ CONCERN: Memory usage higher than expected")
            return False
            
    except Exception as e:
        print(f"✗ ERROR in N={N} test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_deviation_pattern():
    """Test if multiple deviation values create multiple matrix copies"""
    print("\n\n=== MULTIPLE DEVIATION TEST ===")
    print("Testing if processing multiple deviations creates multiple matrix copies")
    
    N = 5000  # Smaller N for this test
    deviations = [0.0, 0.1, 0.2, 0.3, 0.4]  # 5 different deviations
    
    from sqw.experiments_sparse import running_streaming_sparse
    
    print(f"Testing N={N} with {len(deviations)} different deviations")
    
    memory_per_deviation = []
    
    for i, dev in enumerate(deviations):
        print(f"\n  Deviation {i+1}/{len(deviations)}: {dev}")
        
        start_mem = get_memory_usage()
        
        def simple_callback(step_idx, state):
            del state
            gc.collect()
        
        try:
            final_state = running_streaming_sparse(
                N, np.pi/3, 5,  # 5 steps
                initial_nodes=[],
                deviation_range=dev,
                step_callback=simple_callback
            )
            
            peak_mem = get_memory_usage()
            memory_increase = peak_mem - start_mem
            memory_per_deviation.append(memory_increase)
            
            print(f"    Memory increase: {memory_increase:.1f}MB")
            
            del final_state
            gc.collect()
            
        except Exception as e:
            print(f"    ERROR: {e}")
            memory_per_deviation.append(float('inf'))
    
    # Analyze results
    print(f"\n--- MULTIPLE DEVIATION ANALYSIS ---")
    valid_memories = [m for m in memory_per_deviation if m != float('inf')]
    
    if valid_memories:
        avg_memory = sum(valid_memories) / len(valid_memories)
        max_memory = max(valid_memories)
        min_memory = min(valid_memories)
        
        print(f"Memory per deviation:")
        for i, mem in enumerate(memory_per_deviation):
            if mem != float('inf'):
                print(f"  Dev {deviations[i]}: {mem:.1f}MB")
        
        print(f"Average memory per deviation: {avg_memory:.1f}MB")
        print(f"Range: {min_memory:.1f}MB to {max_memory:.1f}MB")
        
        if max_memory - min_memory < avg_memory * 0.2:  # Less than 20% variation
            print("✓ CONSISTENT: All deviations use similar memory")
        else:
            print("⚠ VARIATION: Memory usage varies between deviations")
        
        # Check if memory scales linearly (bad) or stays constant (good)
        if max_memory < avg_memory * 1.5:
            print("✓ GOOD: No evidence of accumulating multiple matrices")
        else:
            print("⚠ CONCERN: Memory usage may be accumulating")

if __name__ == "__main__":
    print("N=20k MEMORY SAFETY VERIFICATION")
    print("=" * 50)
    
    success = test_n20k_memory_safety()
    test_multiple_deviation_pattern()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ VERIFIED: Sparse streaming is memory-safe for N=20k")
        print("   ✓ No 5×20k×20k matrix storage detected")
        print("   ✓ Memory usage consistent with single-matrix streaming")
        print("   ✓ Ready for cluster deployment")
    else:
        print("❌ CONCERN: Memory usage higher than expected")
        print("   Need to investigate potential memory issues")
    print("=" * 50)
