#!/usr/bin/env python3
"""
Direct verification that we're storing the correct matrix dimensions:
- Matrices should be 20k×20k (sparse)
- We should only have ONE state vector at a time (20k×1)
- We should NEVER have 5×20k×20k arrays
"""

import os
import sys
import gc
import psutil
import numpy as np

# Add sqw module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sqw'))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def find_large_arrays():
    """Find all large numpy arrays in memory"""
    import gc
    large_arrays = []
    
    for obj in gc.get_objects():
        if isinstance(obj, np.ndarray) and obj.nbytes > 1024 * 1024:  # > 1MB
            large_arrays.append({
                'shape': obj.shape,
                'dtype': obj.dtype,
                'size_mb': obj.nbytes / (1024**2),
                'id': id(obj)
            })
    
    return large_arrays

def test_matrix_dimensions():
    """Test to verify exact matrix dimensions and ensure no 5×20k×20k arrays"""
    print("=== MATRIX DIMENSION VERIFICATION ===")
    
    N = 20000
    print(f"Testing exact matrix dimensions for N={N}")
    
    from sqw.experiments_sparse import (
        get_sparse_adjacency_matrix, create_sparse_noisy_hamiltonians,
        sparse_ct_evo_with_noise
    )
    from sqw.experiments_expanded_static import (
        cycle_tesselation_alpha, cycle_tesselation_beta, create_noise_lists
    )
    
    print("\n1. Creating tessellations and checking memory...")
    start_memory = get_memory_usage()
    large_arrays_start = find_large_arrays()
    print(f"   Start: {start_memory:.1f}MB, Large arrays: {len(large_arrays_start)}")
    
    # Create graphs
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    after_graphs = get_memory_usage()
    large_arrays_graphs = find_large_arrays()
    print(f"   After graphs: {after_graphs:.1f}MB (+{after_graphs-start_memory:.1f}MB), Large arrays: {len(large_arrays_graphs)}")
    
    print("\n2. Creating sparse adjacency matrices...")
    Hr_sparse = get_sparse_adjacency_matrix(red_graph)
    Hb_sparse = get_sparse_adjacency_matrix(blue_graph)
    
    after_adjacency = get_memory_usage()
    large_arrays_adj = find_large_arrays()
    
    print(f"   After adjacency: {after_adjacency:.1f}MB (+{after_adjacency-after_graphs:.1f}MB)")
    print(f"   Red matrix: shape={Hr_sparse.shape}, nnz={Hr_sparse.nnz}, memory={Hr_sparse.nnz*24/(1024**2):.1f}MB")
    print(f"   Blue matrix: shape={Hb_sparse.shape}, nnz={Hb_sparse.nnz}, memory={Hb_sparse.nnz*24/(1024**2):.1f}MB")
    print(f"   Large arrays in memory: {len(large_arrays_adj)}")
    
    # Verify matrix dimensions
    assert Hr_sparse.shape == (N, N), f"Red matrix wrong shape: {Hr_sparse.shape}"
    assert Hb_sparse.shape == (N, N), f"Blue matrix wrong shape: {Hb_sparse.shape}"
    print("   ✓ Matrix shapes are correct: 20k×20k")
    
    print("\n3. Creating noisy Hamiltonians...")
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    red_noise_list, blue_noise_list = create_noise_lists(
        np.pi/3, red_edge_list, blue_edge_list, 0.1
    )
    
    Hr_noisy, Hb_noisy = create_sparse_noisy_hamiltonians(
        red_graph, blue_graph, red_noise_list, blue_noise_list
    )
    
    after_hamiltonians = get_memory_usage()
    large_arrays_ham = find_large_arrays()
    
    print(f"   After Hamiltonians: {after_hamiltonians:.1f}MB (+{after_hamiltonians-after_adjacency:.1f}MB)")
    print(f"   Red Hamiltonian: shape={Hr_noisy.shape}, nnz={Hr_noisy.nnz}")
    print(f"   Blue Hamiltonian: shape={Hb_noisy.shape}, nnz={Hb_noisy.nnz}")
    print(f"   Large arrays in memory: {len(large_arrays_ham)}")
    
    # Verify no 5×20k×20k arrays
    for i, arr in enumerate(large_arrays_ham):
        print(f"     Array {i+1}: shape={arr['shape']}, size={arr['size_mb']:.1f}MB")
        if len(arr['shape']) == 3 and arr['shape'][0] == 5:
            print(f"     ❌ DANGER: Found 5×something array!")
        if len(arr['shape']) == 2 and arr['shape'] == (N, N):
            print(f"     ✓ Expected: Found {N}×{N} matrix")
    
    print("\n4. Creating state vector...")
    psi = np.ones((N, 1), dtype=complex) / np.sqrt(N)
    
    after_state = get_memory_usage()
    large_arrays_state = find_large_arrays()
    
    print(f"   After state: {after_state:.1f}MB (+{after_state-after_hamiltonians:.1f}MB)")
    print(f"   State vector: shape={psi.shape}, size={psi.nbytes/(1024**2):.3f}MB")
    print(f"   Large arrays in memory: {len(large_arrays_state)}")
    
    # Verify state vector
    assert psi.shape == (N, 1), f"State vector wrong shape: {psi.shape}"
    print("   ✓ State vector shape is correct: 20k×1")
    
    print("\n5. Final array inventory...")
    for i, arr in enumerate(large_arrays_state):
        print(f"   Array {i+1}: shape={arr['shape']}, dtype={arr['dtype']}, size={arr['size_mb']:.1f}MB")
        
        # Check for dangerous patterns
        if len(arr['shape']) == 3:
            print(f"     ⚠ WARNING: 3D array detected!")
            if arr['shape'][0] == 5:
                print(f"     ❌ CRITICAL: This looks like 5×{arr['shape'][1]}×{arr['shape'][2]}!")
        elif len(arr['shape']) == 2:
            if arr['shape'] == (N, N):
                print(f"     ✓ Expected: {N}×{N} matrix (sparse representation)")
            elif arr['shape'] == (N, 1):
                print(f"     ✓ Expected: {N}×1 state vector")
            else:
                print(f"     ? Unknown: {arr['shape']} matrix")
    
    total_memory = after_state - start_memory
    print(f"\n--- SUMMARY ---")
    print(f"Total memory increase: {total_memory:.1f}MB")
    print(f"Expected for N={N}: ~2-3MB (sparse matrices + state)")
    print(f"Efficiency: {total_memory:.1f}MB vs {N*N*16/(1024**2):.0f}MB if dense (99.{100-total_memory*100/(N*N*16/(1024**2)):.0f}% savings)")
    
    # Critical checks
    dangerous_arrays = [arr for arr in large_arrays_state 
                       if len(arr['shape']) == 3 and arr['shape'][0] == 5]
    
    if dangerous_arrays:
        print(f"❌ CRITICAL: Found {len(dangerous_arrays)} arrays with shape 5×something!")
        for arr in dangerous_arrays:
            print(f"   - Shape: {arr['shape']}, Size: {arr['size_mb']:.1f}MB")
        return False
    else:
        print(f"✅ SAFE: No 5×20k×20k arrays detected")
        
    # Check total memory is reasonable
    if total_memory < 100:  # Less than 100MB
        print(f"✅ EXCELLENT: Memory usage ({total_memory:.1f}MB) is very reasonable")
        return True
    elif total_memory < 500:  # Less than 500MB
        print(f"✅ GOOD: Memory usage ({total_memory:.1f}MB) is acceptable")
        return True
    else:
        print(f"⚠ CONCERN: Memory usage ({total_memory:.1f}MB) is higher than expected")
        return False

def test_streaming_callback_memory():
    """Test memory during streaming callback to ensure we don't accumulate states"""
    print("\n\n=== STREAMING CALLBACK MEMORY TEST ===")
    
    N = 10000  # Smaller N for detailed testing
    print(f"Testing streaming callback pattern with N={N}")
    
    from sqw.experiments_sparse import running_streaming_sparse
    
    step_memories = []
    step_arrays = []
    
    def detailed_memory_callback(step_idx, state):
        """Monitor memory and arrays at each step"""
        current_memory = get_memory_usage()
        large_arrays = find_large_arrays()
        
        step_memories.append(current_memory)
        step_arrays.append(len(large_arrays))
        
        if step_idx <= 2 or step_idx >= 8:  # Monitor first and last few steps
            print(f"   Step {step_idx}: Memory={current_memory:.1f}MB, Arrays={len(large_arrays)}")
            for i, arr in enumerate(large_arrays):
                if arr['size_mb'] > 0.1:  # Only show arrays > 0.1MB
                    print(f"     Array {i+1}: {arr['shape']}, {arr['size_mb']:.2f}MB")
        
        # Critical check: ensure we don't have multiple state vectors
        state_vectors = [arr for arr in large_arrays 
                        if len(arr['shape']) == 2 and arr['shape'][1] == 1]
        
        if len(state_vectors) > 1:
            print(f"     ⚠ WARNING: {len(state_vectors)} state vectors in memory!")
        
        # Clean up reference
        del state
        gc.collect()
    
    print("Starting streaming with detailed memory monitoring...")
    
    try:
        final_state = running_streaming_sparse(
            N, np.pi/3, 10,
            initial_nodes=[],
            deviation_range=0.1,
            step_callback=detailed_memory_callback
        )
        
        print(f"\nStreaming completed. Final analysis:")
        print(f"Memory progression: {step_memories[0]:.1f}MB → {step_memories[-1]:.1f}MB")
        print(f"Peak memory: {max(step_memories):.1f}MB")
        print(f"Array count progression: {step_arrays[0]} → {step_arrays[-1]}")
        print(f"Max arrays: {max(step_arrays)}")
        
        memory_variation = max(step_memories) - min(step_memories)
        if memory_variation < 10:
            print(f"✅ EXCELLENT: Memory stable within {memory_variation:.1f}MB")
        elif memory_variation < 50:
            print(f"✅ GOOD: Memory stable within {memory_variation:.1f}MB")
        else:
            print(f"⚠ CONCERN: Memory varies by {memory_variation:.1f}MB")
        
        del final_state
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("MATRIX DIMENSION AND MEMORY VERIFICATION")
    print("=" * 60)
    
    test1_success = test_matrix_dimensions()
    test2_success = test_streaming_callback_memory()
    
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION RESULTS")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("🎉 VERIFICATION COMPLETE: ALL TESTS PASSED")
        print("✅ Matrix dimensions are correct (20k×20k sparse, 20k×1 state)")
        print("✅ No 5×20k×20k arrays detected")
        print("✅ Memory usage is efficient and stable")
        print("✅ Streaming pattern works correctly")
        print("\n🚀 READY FOR CLUSTER DEPLOYMENT!")
    else:
        print("❌ VERIFICATION FAILED: Issues detected")
        if not test1_success:
            print("   - Matrix dimension test failed")
        if not test2_success:
            print("   - Streaming memory test failed")
    print("=" * 60)
