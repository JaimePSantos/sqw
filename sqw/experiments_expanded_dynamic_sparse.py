"""
SPARSE MATRIX OPTIMIZED Dynamic Quantum Walk Implementation

This addresses the O(N²) scaling issue by using sparse matrices
and optimized operations that scale better with system size.

Key optimizations:
1. Use sparse matrices throughout
2. Avoid full N×N matrix operations where possible
3. Leverage the structure of the tesselations
4. Use more efficient evolution strategies
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import expm_multiply
import time

def build_sparse_hamiltonian(graph, tesselation, matrix_representation='adjacency'):
    """Build Hamiltonian using sparse matrices"""
    if matrix_representation == 'adjacency':
        matrix = nx.adjacency_matrix(graph, format='csr')
    elif matrix_representation == 'laplacian':
        matrix = nx.laplacian_matrix(graph, format='csr')
    
    num_nodes = graph.number_of_nodes()
    
    # For line graphs with simple tesselations, we can build the Hamiltonian more efficiently
    # Each tesselation only affects adjacent pairs
    from itertools import combinations
    
    # Start with zero sparse matrix
    hamiltonian = csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    
    for tesselation_subset in tesselation:
        edge_combinations = combinations(tesselation_subset, 2)
        
        for edge in edge_combinations:
            node_i, node_j = edge[0], edge[1]
            # Add only the non-zero elements
            hamiltonian[node_i, node_j] = matrix[node_i, node_j]
            hamiltonian[node_j, node_i] = matrix[node_i, node_j]
            hamiltonian[node_i, node_i] = matrix[node_i, node_i]
            hamiltonian[node_j, node_j] = matrix[node_j, node_j]
    
    return hamiltonian

def running_streaming_dynamic_sparse(graph, tesselation_list, num_steps, 
                                   initial_state, angles, tesselation_order,
                                   matrix_representation='adjacency',
                                   searching=[], step_callback=None, logger=None):
    """
    Sparse matrix implementation for better scaling
    
    This version uses sparse matrices to reduce the O(N²) scaling
    """
    
    if logger:
        logger.debug("Building sparse Hamiltonians...")
    hamiltonians = []
    
    # Build sparse Hamiltonians
    for tesselation in tesselation_list:
        h = build_sparse_hamiltonian(graph, tesselation, matrix_representation)
        hamiltonians.append(h)
    
    num_nodes = graph.number_of_nodes()
    num_tesselations = len(tesselation_list)

    # Save initial state if callback provided
    if step_callback:
        step_callback(0, initial_state.copy())

    if logger:
        logger.debug("Starting sparse matrix evolution...")
    current_state = np.array(initial_state, dtype=np.complex128).flatten()
    
    for time_step in range(num_steps):
        # For sparse matrices, we can use expm_multiply which is more efficient
        # than computing the full matrix exponential
        
        # Apply each tesselation evolution
        for tesselation_idx in tesselation_order[time_step]:
            angle = angles[time_step][tesselation_idx]
            hamiltonian = hamiltonians[tesselation_idx]
            
            # Use sparse matrix exponential multiplication
            # This scales much better than full matrix operations
            current_state = expm_multiply(-1j * angle * hamiltonian, current_state)
        
        # Apply search operator if needed
        if searching:
            for search_node in searching:
                current_state[search_node] *= -1
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(time_step + 1, current_state.copy())
        
        # Progress reporting
        if (time_step + 1) % 1000 == 0 or time_step == num_steps - 1:
            if logger:
                logger.debug(f"Evolution step {time_step + 1}/{num_steps} completed")
                   
    return current_state

def running_streaming_dynamic_optimized_structure(graph, tesselation_list, num_steps, 
                                                 initial_state, angles, tesselation_order,
                                                 matrix_representation='adjacency',
                                                 searching=[], step_callback=None, logger=None):
    """
    Optimized implementation that leverages the specific structure of line graphs
    
    For line graphs, we can avoid many matrix operations by using the specific
    structure of the evolution operators.
    """
    
    num_nodes = graph.number_of_nodes()
    
    # For line graphs, tesselations have a very specific structure
    # We can optimize based on this
    
    if logger:
        logger.info("Starting structure-optimized evolution...")
    current_state = np.array(initial_state, dtype=np.complex128).flatten()
    
    # Save initial state if callback provided
    if step_callback:
        step_callback(0, initial_state.copy())
    
    for time_step in range(num_steps):
        # For line graphs with simple tesselations, we can apply
        # the evolution more efficiently by working with pairs
        
        # Apply each tesselation
        for tesselation_idx in tesselation_order[time_step]:
            angle = angles[time_step][tesselation_idx]
            tesselation = tesselation_list[tesselation_idx]
            
            # For each pair in the tesselation, apply a 2x2 rotation
            for pair in tesselation:
                if len(pair) == 2:
                    i, j = pair[0], pair[1]
                    
                    # Extract the 2x2 subspace
                    old_i = current_state[i]
                    old_j = current_state[j]
                    
                    # Apply 2x2 rotation (much faster than full N×N)
                    cos_a = np.cos(angle)
                    sin_a = np.sin(angle)
                    
                    current_state[i] = cos_a * old_i - 1j * sin_a * old_j
                    current_state[j] = cos_a * old_j - 1j * sin_a * old_i
        
        # Apply search operator if needed
        if searching:
            for search_node in searching:
                current_state[search_node] *= -1
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(time_step + 1, current_state.copy())
        
        # Progress reporting
        if (time_step + 1) % 1000 == 0 or time_step == num_steps - 1:
            if logger:
                logger.info(f"Evolution step {time_step + 1}/{num_steps} completed")
    
    return current_state

def compare_implementations_scaling():
    """Compare different implementations for scaling"""
    # Disabled to avoid I/O errors in cluster environments
    return
    
    # The rest of this function is commented out to prevent print() I/O errors
    sizes = [100, 500, 1000]
    
    for N in sizes:
        steps = 50  # Fixed steps for fair comparison
        print(f"\nTesting N={N}, steps={steps}:")
        
        # Setup
        import networkx as nx
        from sqw.tesselations import even_line_two_tesselation
        from sqw.states import uniform_initial_state
        
        graph = nx.path_graph(N)
        tesselation = even_line_two_tesselation(N)
        initial_state = uniform_initial_state(N, nodes=[N//2])
        angles = [[np.pi/3, np.pi/3] for _ in range(steps)]
        tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
        
        # Test original eigenvalue implementation
        try:
            from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
            
            start_time = time.time()
            final_state_original = running_streaming_dynamic_eigenvalue_based(
                graph, tesselation, steps, initial_state, angles, tesselation_order,
                matrix_representation='adjacency', searching=[]
            )
            original_time = time.time() - start_time
            print(f"  Original eigenvalue: {original_time:.3f}s ({original_time/steps:.4f}s/step)")
            
        except Exception as e:
            print(f"  Original failed: {e}")
            original_time = None
            final_state_original = None
        
        # Test sparse implementation
        try:
            start_time = time.time()
            final_state_sparse = running_streaming_dynamic_sparse(
                graph, tesselation, steps, initial_state, angles, tesselation_order,
                matrix_representation='adjacency', searching=[]
            )
            sparse_time = time.time() - start_time
            print(f"  Sparse matrix: {sparse_time:.3f}s ({sparse_time/steps:.4f}s/step)")
            
            if original_time:
                speedup = original_time / sparse_time
                print(f"  Sparse speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Sparse failed: {e}")
            sparse_time = None
            final_state_sparse = None
        
        # Test structure-optimized implementation
        try:
            start_time = time.time()
            final_state_struct = running_streaming_dynamic_optimized_structure(
                graph, tesselation, steps, initial_state, angles, tesselation_order,
                matrix_representation='adjacency', searching=[]
            )
            struct_time = time.time() - start_time
            print(f"  Structure-optimized: {struct_time:.3f}s ({struct_time/steps:.4f}s/step)")
            
            if original_time:
                speedup = original_time / struct_time
                print(f"  Structure speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"  Structure-optimized failed: {e}")
            struct_time = None
            final_state_struct = None
        
        # Estimate production scaling
        if struct_time:
            # Assume O(N) scaling for structure-optimized version
            prod_scaling = (20000/N) * (5000/steps)
            prod_estimate = struct_time * prod_scaling
            print(f"  Production estimate (structure): {prod_estimate:.1f}s ({prod_estimate/60:.1f}min)")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    compare_implementations_scaling()
