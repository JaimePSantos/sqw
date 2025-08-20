#!/usr/bin/env python3
"""
Memory profiling test to verify matrix sizes and memory usage patterns.
This will test different N values and confirm we're not storing 5x20kx20k matrices.
"""

import os
import sys
import gc
import psutil
import numpy as np
import time
import tracemalloc

# Add sqw module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sqw'))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)

def get_matrix_info(matrix):
    """Get detailed information about a matrix"""
    if hasattr(matrix, 'shape'):
        shape = matrix.shape
        if hasattr(matrix, 'nnz'):  # Sparse matrix
            nnz = matrix.nnz
            sparsity = nnz / (shape[0] * shape[1]) if shape[0] * shape[1] > 0 else 0
            memory_mb = (nnz * 16 + nnz * 8) / (1024**2)  # 16 bytes data + 8 bytes indices approx
            return f"Sparse {shape}, nnz={nnz}, sparsity={sparsity:.4f}, ~{memory_mb:.1f}MB"
        else:  # Dense matrix
            total_elements = np.prod(shape)
            memory_mb = total_elements * matrix.itemsize / (1024**2)
            return f"Dense {shape}, {total_elements} elements, {memory_mb:.1f}MB"
    return f"Unknown matrix type: {type(matrix)}"

def test_matrix_creation_memory():
    """Test memory usage during matrix creation for different N values"""
    print("=== MATRIX CREATION MEMORY TEST ===")
    
    # Import required modules
    from sqw.experiments_expanded_static import cycle_tesselation_alpha, cycle_tesselation_beta
    from sqw.experiments_sparse import get_sparse_adjacency_matrix, create_sparse_noisy_hamiltonians
    
    test_sizes = [100, 1000, 5000, 10000, 20000]
    
    for N in test_sizes:
        print(f"\n--- Testing N={N} ---")
        
        # Start memory tracking
        start_memory = get_memory_usage()
        tracemalloc.start()
        
        try:
            # Create tessellations
            print(f"  Creating tessellations...")
            red_graph = cycle_tesselation_alpha(N)
            blue_graph = cycle_tesselation_beta(N)
            
            after_graphs = get_memory_usage()
            print(f"  After graphs: {after_graphs - start_memory:+.1f}MB")
            
            # Get sparse adjacency matrices
            print(f"  Creating sparse adjacency matrices...")
            Hr_sparse = get_sparse_adjacency_matrix(red_graph)
            Hb_sparse = get_sparse_adjacency_matrix(blue_graph)
            
            after_adjacency = get_memory_usage()
            print(f"  After adjacency: {after_adjacency - after_graphs:+.1f}MB")
            print(f"    Red matrix: {get_matrix_info(Hr_sparse)}")
            print(f"    Blue matrix: {get_matrix_info(Hb_sparse)}")
            
            # Create noise lists (small)
            red_edge_list = list(red_graph.edges())
            blue_edge_list = list(blue_graph.edges())
            
            red_noise_list = [np.pi/3 + 0.1 for _ in red_edge_list]
            blue_noise_list = [np.pi/3 - 0.1 for _ in blue_edge_list]
            
            print(f"  Noise lists: {len(red_noise_list)} red edges, {len(blue_noise_list)} blue edges")
            
            # Create noisy Hamiltonians
            print(f"  Creating noisy Hamiltonians...")
            Hr_noisy, Hb_noisy = create_sparse_noisy_hamiltonians(
                red_graph, blue_graph, red_noise_list, blue_noise_list
            )
            
            after_hamiltonians = get_memory_usage()
            print(f"  After Hamiltonians: {after_hamiltonians - after_adjacency:+.1f}MB")
            print(f"    Red Hamiltonian: {get_matrix_info(Hr_noisy)}")
            print(f"    Blue Hamiltonian: {get_matrix_info(Hb_noisy)}")
            
            # Check if we can create evolution operators for smaller N
            if N <= 5000:  # Only test evolution for smaller matrices
                print(f"  Creating evolution operators...")
                from scipy.sparse.linalg import expm as sparse_expm
                
                R = sparse_expm(1j * Hr_noisy)
                after_R = get_memory_usage()
                print(f"  After R operator: {after_R - after_hamiltonians:+.1f}MB")
                print(f"    R operator: {get_matrix_info(R)}")
                
                B = sparse_expm(1j * Hb_noisy)
                after_B = get_memory_usage()
                print(f"  After B operator: {after_B - after_R:+.1f}MB")
                print(f"    B operator: {get_matrix_info(B)}")
                
                U = B @ R
                after_U = get_memory_usage()
                print(f"  After U operator: {after_U - after_B:+.1f}MB")
                print(f"    U operator: {get_matrix_info(U)}")
                
                # Test state vector
                print(f"  Creating state vector...")
                psi = np.ones((N, 1), dtype=complex) / np.sqrt(N)
                after_state = get_memory_usage()
                print(f"  After state: {after_state - after_U:+.1f}MB")
                print(f"    State vector: {get_matrix_info(psi)}")
                
                total_memory = after_state - start_memory
                print(f"  TOTAL MEMORY for N={N}: {total_memory:.1f}MB")
                
                # Clean up evolution operators
                del R, B, U, psi
            else:
                total_memory = after_hamiltonians - start_memory
                print(f"  TOTAL MEMORY for N={N} (no evolution): {total_memory:.1f}MB")
                print(f"  Evolution operators not created for N={N} (would be too large)")
            
            # Memory snapshot
            current, peak = tracemalloc.get_traced_memory()
            print(f"  TraceMalloc: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
            
        except Exception as e:
            print(f"  ERROR for N={N}: {e}")
            if N == 20000:
                print(f"  This might be expected for N=20000 if memory is insufficient")
        
        finally:
            # Clean up
            tracemalloc.stop()
            gc.collect()
            
        print(f"  Final memory after cleanup: {get_memory_usage():.1f}MB")

def test_streaming_memory_pattern():
    """Test the streaming pattern to verify we don't accumulate multiple states"""
    print("\n\n=== STREAMING MEMORY PATTERN TEST ===")
    
    from sqw.experiments_sparse import running_streaming_sparse
    
    N = 1000  # Use smaller N for detailed monitoring
    steps = 10
    
    print(f"Testing streaming pattern: N={N}, steps={steps}")
    
    # Memory tracking arrays
    memory_at_steps = []
    state_sizes = []
    
    def detailed_callback(step_idx, state):
        """Callback that monitors memory usage in detail"""
        current_memory = get_memory_usage()
        state_size = state.size * state.itemsize / (1024**2)
        
        memory_at_steps.append(current_memory)
        state_sizes.append(state_size)
        
        print(f"    Step {step_idx:2d}: Memory={current_memory:6.1f}MB, State={state_size:.3f}MB")
        
        # Verify state properties
        if step_idx == 0:
            print(f"      State details: shape={state.shape}, dtype={state.dtype}")
            prob_sum = np.sum(np.abs(state)**2)
            print(f"      Probability sum: {prob_sum:.6f}")
        
        # Clean up as the real callback does
        del state
        gc.collect()
    
    print("Starting streaming computation with detailed monitoring...")
    start_memory = get_memory_usage()
    
    try:
        final_state = running_streaming_sparse(
            N, np.pi/3, steps,
            initial_nodes=[],
            deviation_range=0.1,
            step_callback=detailed_callback
        )
        
        end_memory = get_memory_usage()
        
        print(f"\nStreaming test completed:")
        print(f"  Start memory: {start_memory:.1f}MB")
        print(f"  End memory: {end_memory:.1f}MB")
        print(f"  Memory change: {end_memory - start_memory:+.1f}MB")
        
        # Analyze memory pattern
        if memory_at_steps:
            min_mem = min(memory_at_steps)
            max_mem = max(memory_at_steps)
            avg_mem = sum(memory_at_steps) / len(memory_at_steps)
            
            print(f"  During computation:")
            print(f"    Min memory: {min_mem:.1f}MB")
            print(f"    Max memory: {max_mem:.1f}MB")
            print(f"    Avg memory: {avg_mem:.1f}MB")
            print(f"    Variation: {max_mem - min_mem:.1f}MB ({100*(max_mem-min_mem)/avg_mem:.1f}%)")
            
            # Check if memory is stable (good streaming)
            if (max_mem - min_mem) / avg_mem < 0.1:
                print(f"  ✓ EXCELLENT: Memory usage is very stable (<10% variation)")
            elif (max_mem - min_mem) / avg_mem < 0.3:
                print(f"  ✓ GOOD: Memory usage is stable (<30% variation)")
            else:
                print(f"  ⚠ WARNING: Memory usage varies significantly")
        
        # Verify state sizes are consistent
        if state_sizes:
            expected_state_size = N * 16 / (1024**2)  # Complex128 = 16 bytes per element
            avg_state_size = sum(state_sizes) / len(state_sizes)
            print(f"  State size analysis:")
            print(f"    Expected: {expected_state_size:.3f}MB")
            print(f"    Average observed: {avg_state_size:.3f}MB")
            print(f"    All states same size: {len(set(f'{s:.3f}' for s in state_sizes)) == 1}")
        
        del final_state
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR in streaming test: {e}")
        import traceback
        traceback.print_exc()

def test_large_n_memory_estimate():
    """Test memory estimates for large N without actually creating matrices"""
    print("\n\n=== LARGE N MEMORY ESTIMATES ===")
    
    large_n_values = [10000, 20000, 50000, 100000]
    
    for N in large_n_values:
        print(f"\n--- N={N} Memory Estimates ---")
        
        # For cycle graphs, each node connects to 2 neighbors
        edges_per_graph = N  # Approximately N edges in a cycle
        
        # Sparse matrix memory (approximate)
        # Each non-zero: 8 bytes data + 4 bytes row index + 4 bytes col index
        sparse_matrix_mb = (edges_per_graph * 2 * 16) / (1024**2)  # *2 for symmetric, 16 bytes per entry
        
        # State vector memory
        state_mb = N * 16 / (1024**2)  # Complex128
        
        # Total estimated memory for streaming
        total_mb = 2 * sparse_matrix_mb + state_mb  # 2 Hamiltonians + 1 state
        
        print(f"  Estimated sparse matrix size: {sparse_matrix_mb:.1f}MB each")
        print(f"  Estimated state vector size: {state_mb:.1f}MB")
        print(f"  Estimated total memory: {total_mb:.1f}MB")
        
        # Compare with dense matrix (if we were using dense)
        dense_matrix_mb = N * N * 16 / (1024**2)
        print(f"  Dense matrix would be: {dense_matrix_mb:.1f}MB each")
        print(f"  Sparse memory savings: {100 * (1 - sparse_matrix_mb/dense_matrix_mb):.1f}%")
        
        # Check if it fits in typical cluster memory
        if total_mb < 1000:  # 1GB
            print(f"  ✓ EXCELLENT: Fits easily in 1GB")
        elif total_mb < 4000:  # 4GB
            print(f"  ✓ GOOD: Fits in 4GB")
        elif total_mb < 16000:  # 16GB
            print(f"  ⚠ CAUTION: Needs 16GB+ memory")
        else:
            print(f"  ✗ WARNING: Requires {total_mb/1024:.1f}GB memory")

if __name__ == "__main__":
    print("COMPREHENSIVE MEMORY AND MATRIX SIZE ANALYSIS")
    print("=" * 60)
    
    print(f"System info:")
    print(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    print(f"  CPU count: {psutil.cpu_count()}")
    
    # Run all tests
    test_matrix_creation_memory()
    test_streaming_memory_pattern()
    test_large_n_memory_estimate()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
