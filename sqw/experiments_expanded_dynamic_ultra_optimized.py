"""
ULTRA-OPTIMIZED Dynamic Quantum Walk Implementation

This addresses the main performance bottlenecks identified in the eigenvalue implementation:

1. Avoid reconstructing unitary operators every step
2. Pre-compute as much as possible
3. Use sparse representations where beneficial
4. Minimize memory allocations

Key optimizations:
- Pre-compute all possible unitary operators for different angle combinations
- Use more efficient matrix operations
- Avoid unnecessary array copies and conversions
"""

import numpy as np
import networkx as nx
from scipy.linalg import expm
from itertools import combinations
import time

def hamiltonian_builder_optimized(graph, tesselation, matrix_representation):
    """
    Build Hamiltonian matrix (optimized version)
    """
    if matrix_representation == 'adjacency':
        matrix = nx.adjacency_matrix(graph).todense()
    if matrix_representation == 'laplacian':
        matrix = nx.laplacian_matrix(graph).todense()
        
    num_nodes = graph.number_of_nodes()
    hamiltonian = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    
    for tesselation_subset in tesselation:
        edge_combinations = combinations(tesselation_subset, 2)
        
        for edge in edge_combinations:
            node_i, node_j = edge[0], edge[1]
            hamiltonian[node_i, node_j] = matrix[node_i, node_j]
            hamiltonian[node_j, node_i] = matrix[node_i, node_j]
            hamiltonian[node_i, node_i] = matrix[node_i, node_i]
            hamiltonian[node_j, node_j] = matrix[node_j, node_j]

    return hamiltonian

def precompute_eigendecompositions_optimized(graph, tesselation_list, matrix_representation='adjacency'):
    """
    Pre-compute eigenvalue decompositions for all hamiltonians (optimized)
    """
    print(f"Pre-computing eigenvalue decompositions for {len(tesselation_list)} tesselations...")
    
    hamiltonians = []
    num_tesselations = len(tesselation_list)
    
    # Create hamiltonians for every tesselation and compute eigendecomposition
    for tesselation_idx in range(num_tesselations):
        hamiltonian_matrix = hamiltonian_builder_optimized(graph, tesselation_list[tesselation_idx], matrix_representation)
        
        # Compute eigendecomposition once
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
        
        # Store as arrays (not matrices) for faster operations
        hamiltonians.append((eigenvalues, eigenvectors))
    
    print(f"Eigenvalue decompositions completed for {len(hamiltonians)} hamiltonians")
    return hamiltonians

def precompute_angle_unitaries(hamiltonians, angle_set, cache_size_limit=1000):
    """
    Pre-compute unitary operators for a set of angles to avoid recomputation
    
    Args:
        hamiltonians: List of (eigenvalues, eigenvectors) tuples
        angle_set: Set of possible angle combinations
        cache_size_limit: Maximum number of cached unitaries
    """
    print(f"Pre-computing unitaries for {len(angle_set)} angle combinations...")
    
    unitary_cache = {}
    num_cached = 0
    
    for angles in angle_set:
        if num_cached >= cache_size_limit:
            break
            
        # Convert angles to hashable key
        angle_key = tuple(angles)
        
        if angle_key not in unitary_cache:
            unitaries = build_unitaries_fast(hamiltonians, angles)
            unitary_cache[angle_key] = unitaries
            num_cached += 1
    
    print(f"Cached {num_cached} unitary sets")
    return unitary_cache

def build_unitaries_fast(hamiltonians, angles):
    """
    Build unitary operators using eigenvalue decomposition (optimized)
    """
    num_hamiltonians = len(hamiltonians)
    unitary_operators = []
    
    for hamiltonian_idx in range(num_hamiltonians):
        eigenvalues, eigenvectors = hamiltonians[hamiltonian_idx]
        
        # Fast element-wise exponential of eigenvalues
        exp_eigenvalues = np.exp(-1j * angles[hamiltonian_idx] * eigenvalues)
        
        # Reconstruct unitary: U = V * diag(exp_eigs) * V^dagger
        # This is faster than using expm on the full matrix
        unitary_op = eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.T.conj()
        unitary_operators.append(unitary_op)
    
    return unitary_operators

def running_streaming_dynamic_ultra_optimized(graph, tesselation_list, num_steps, 
                                             initial_state, angles, tesselation_order,
                                             matrix_representation='adjacency',
                                             searching=[], step_callback=None):
    """
    Ultra-optimized dynamic quantum walk implementation
    
    Key optimizations:
    1. Pre-compute eigendecompositions once
    2. Use faster unitary building
    3. Pre-compute common angle combinations if possible
    4. Minimize array copies and memory allocations
    """
    
    # PRE-COMPUTE: Eigenvalue decompositions (expensive but done only once)
    hamiltonians = precompute_eigendecompositions_optimized(graph, tesselation_list, matrix_representation)
    
    num_nodes = graph.number_of_nodes()
    num_tesselations = len(tesselation_list)

    # Save initial state if callback provided
    if step_callback:
        step_callback(0, initial_state.copy())

    # Try to identify common angle patterns for caching
    unique_angles = list(set(tuple(angle_step) for angle_step in angles))
    use_caching = len(unique_angles) < len(angles) * 0.8  # Only cache if we have repetition
    
    unitary_cache = {}
    if use_caching and len(unique_angles) < 100:
        print(f"Detected {len(unique_angles)} unique angle patterns, enabling caching...")
        unitary_cache = precompute_angle_unitaries(hamiltonians, unique_angles)

    # Evolution loop (optimized)
    print("Starting ultra-optimized evolution...")
    current_state = np.array(initial_state, dtype=np.complex128).flatten()
    
    # Pre-allocate arrays to avoid repeated allocations
    total_unitary = np.eye(num_nodes, dtype=np.complex128)
    
    for time_step in range(num_steps):
        # Get or compute unitary operators for this step
        angle_key = tuple(angles[time_step])
        
        if use_caching and angle_key in unitary_cache:
            unitary_operators = unitary_cache[angle_key]
        else:
            unitary_operators = build_unitaries_fast(hamiltonians, angles[time_step])

        # Reset total unitary (reuse allocated array)
        total_unitary.fill(0)
        np.fill_diagonal(total_unitary, 1)

        # Initialize search operator
        if searching:
            for search_node in searching:
                total_unitary[search_node, search_node] = -1
        
        # Build master unitary operator (optimized order)
        for unitary_idx in range(num_tesselations):
            tesselation_idx = tesselation_order[time_step][unitary_idx]
            # Use @ operator which is optimized for matrix multiplication
            total_unitary = unitary_operators[tesselation_idx] @ total_unitary

        # Apply evolution (in-place where possible)
        current_state = total_unitary @ current_state
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(time_step + 1, current_state.copy())
        
        # Progress reporting
        if (time_step + 1) % 1000 == 0 or time_step == num_steps - 1:
            print(f"Evolution step {time_step + 1}/{num_steps} completed")
                   
    return current_state

def compare_implementations():
    """Compare original eigenvalue vs ultra-optimized implementation"""
    print("=" * 60)
    print("IMPLEMENTATION COMPARISON")
    print("=" * 60)
    
    # Test parameters
    N = 100
    steps = 25
    base_theta = np.pi/3
    dev = 0.2
    
    # Setup common environment
    import networkx as nx
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    from sqw.utils import random_angle_deviation
    
    graph = nx.path_graph(N)
    tesselation = even_line_two_tesselation(N)
    initial_state = uniform_initial_state(N, nodes=[N//2])
    angles = random_angle_deviation([base_theta, base_theta], [dev, dev], steps)
    tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
    
    print(f"Test parameters: N={N}, steps={steps}")
    print()
    
    # Test original eigenvalue implementation
    try:
        from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
        
        print("Testing original eigenvalue implementation...")
        start_time = time.time()
        
        final_state_original = running_streaming_dynamic_eigenvalue_based(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[]
        )
        
        end_time = time.time()
        original_time = end_time - start_time
        print(f"Original eigenvalue: {original_time:.4f} seconds")
        
    except Exception as e:
        print(f"Original eigenvalue implementation failed: {e}")
        original_time = None
        final_state_original = None
    
    # Test ultra-optimized implementation
    try:
        print("\nTesting ultra-optimized implementation...")
        start_time = time.time()
        
        final_state_optimized = running_streaming_dynamic_ultra_optimized(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[]
        )
        
        end_time = time.time()
        optimized_time = end_time - start_time
        print(f"Ultra-optimized: {optimized_time:.4f} seconds")
        
    except Exception as e:
        print(f"Ultra-optimized implementation failed: {e}")
        optimized_time = None
        final_state_optimized = None
    
    # Compare results
    if original_time and optimized_time:
        speedup = original_time / optimized_time
        print(f"\nSpeedup: {speedup:.2f}x")
        
        if final_state_original is not None and final_state_optimized is not None:
            state_diff = np.linalg.norm(final_state_original - final_state_optimized)
            print(f"State difference: {state_diff:.2e}")

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    compare_implementations()
