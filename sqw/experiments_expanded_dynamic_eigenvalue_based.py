"""
EIGENVALUE-BASED Dynamic Quantum Walk Implementation

This module follows the original experiments_expanded.py approach:
1. Compute eigenvalue decomposition once per hamiltonian
2. Use element-wise exponential of eigenvalues (very fast)
3. Should match the performance of the original implementation

This is the correct optimization based on the original code structure.
"""

import numpy as np
import networkx as nx
from scipy.linalg import expm
from itertools import combinations


def hamiltonian_builder_dynamic(graph, tesselation, matrix_representation):
    """
    Build Hamiltonian matrix (identical to original experiments_expanded.py)
    """
    if matrix_representation == 'adjacency':
        matrix = nx.adjacency_matrix(graph).todense()
    if matrix_representation == 'laplacian':
        matrix = nx.laplacian_matrix(graph).todense()
        
    num_nodes = graph.number_of_nodes()
    hamiltonian = np.zeros((num_nodes, num_nodes))
    
    for tesselation_subset in tesselation:
        edge_combinations = combinations(tesselation_subset, 2)
        
        for edge in edge_combinations:
            node_i, node_j = edge[0], edge[1]
            hamiltonian[node_i, node_j] = matrix[node_i, node_j]
            hamiltonian[node_j, node_i] = matrix[node_i, node_j]
            hamiltonian[node_i, node_i] = matrix[node_i, node_i]
            hamiltonian[node_j, node_j] = matrix[node_j, node_j]

    return hamiltonian


def precompute_eigendecompositions(graph, tesselation_list, matrix_representation='adjacency'):
    """
    Pre-compute eigenvalue decompositions for all hamiltonians.
    This is the key optimization from the original code!
    """
    print(f"Pre-computing eigenvalue decompositions for {len(tesselation_list)} tesselations...")
    
    hamiltonians = []
    num_tesselations = len(tesselation_list)
    
    # Create hamiltonians for every tesselation and compute eigendecomposition
    for tesselation_idx in range(num_tesselations):
        hamiltonian_matrix = hamiltonian_builder_dynamic(graph, tesselation_list[tesselation_idx], matrix_representation)
        
        # This is the expensive part - but we do it only once per hamiltonian!
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
        hamiltonians.append((np.diag(eigenvalues), np.matrix(eigenvectors)))
    
    print(f"Eigenvalue decompositions completed for {len(hamiltonians)} hamiltonians")
    return hamiltonians


def unitary_builder_dynamic(hamiltonians, rotation_angles):
    """
    Build unitary operators using eigenvalue decomposition (identical to original)
    This is very fast because we work with diagonal eigenvalue matrices!
    """
    num_hamiltonians = len(hamiltonians)
    unitary_operators = []
    
    for hamiltonian_idx in range(num_hamiltonians):
        eigenvalues_matrix, eigenvectors_matrix = hamiltonians[hamiltonian_idx]
        
        # This is FAST: expm of diagonal matrix = element-wise exp!
        evolution_operator = expm(-1j * rotation_angles[hamiltonian_idx] * eigenvalues_matrix)
        
        # Reconstruct full unitary operator - convert back to array for proper matrix multiplication
        unitary_op = np.array(eigenvectors_matrix @ evolution_operator @ eigenvectors_matrix.H)
        unitary_operators.append(unitary_op)
    
    return unitary_operators


def running_streaming_dynamic_eigenvalue_based(graph, tesselation_list, num_steps, 
                                             initial_state, angles, tesselation_order,
                                             matrix_representation='adjacency',
                                             searching=[], step_callback=None):
    """
    Dynamic quantum walk using eigenvalue decomposition approach from original experiments_expanded.py.
    
    This should be much faster because:
    1. Eigendecomposition is done once per hamiltonian (not per step)
    2. Matrix exponential becomes element-wise exponential (very fast)
    3. Follows the exact structure of the original fast implementation
    """
    
    # PRE-COMPUTE: Eigenvalue decompositions (expensive but done only once)
    hamiltonians = precompute_eigendecompositions(graph, tesselation_list, matrix_representation)
    
    num_nodes = graph.number_of_nodes()
    num_tesselations = len(tesselation_list)

    # Save initial state if callback provided
    if step_callback:
        step_callback(0, initial_state.copy())

    # Evolution loop (this should be very fast now!)
    print("Starting evolution with eigenvalue-based approach...")
    current_state = np.array(initial_state).flatten()  # Ensure proper shape
    
    for time_step in range(num_steps):
        # Build unitary operators for this step using eigenvalue decomposition
        # This is FAST because we use pre-computed eigendecompositions!
        unitary_operators = unitary_builder_dynamic(hamiltonians, angles[time_step])

        # Initialize search operator
        if searching == []:
            total_unitary = np.eye(num_nodes, dtype='complex')
        else: 
            total_unitary = np.eye(num_nodes, dtype='complex')
            for search_node in searching:
                total_unitary[search_node, search_node] = -1
        
        # Build master unitary operator
        for unitary_idx in range(num_tesselations):
            tesselation_idx = tesselation_order[time_step][unitary_idx]
            total_unitary = np.array(unitary_operators[tesselation_idx]) @ total_unitary

        # Apply evolution - ensure both are proper arrays
        current_state = total_unitary @ current_state.flatten()
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(time_step + 1, current_state.copy())
        
        # Progress reporting
        if (time_step + 1) % 100 == 0 or time_step == num_steps - 1:
            print(f"Evolution step {time_step + 1}/{num_steps} completed")
                   
    return current_state


def test_eigenvalue_based_performance():
    """Test the eigenvalue-based implementation"""
    import time
    
    # Test parameters
    N = 100
    steps = 25
    
    # Create test setup
    G = nx.cycle_graph(N)
    from sqw.tesselations import even_line_two_tesselation
    T = even_line_two_tesselation(N)
    
    # Create test angles
    base_theta = np.pi/3
    angles = [[base_theta, base_theta]] * steps
    tesselation_order = [[0, 1] for _ in range(steps)]
    
    # Create initial state
    initial_state = np.zeros(N, dtype=np.complex128)
    initial_state[N//2] = 1.0
    
    print("Testing EIGENVALUE-BASED dynamic implementation...")
    start_time = time.time()
    
    def dummy_callback(step, state):
        if step % 5 == 0:
            print(f"  Evolution step {step}/{steps} completed")
    
    final_state = running_streaming_dynamic_eigenvalue_based(
        G, T, steps, initial_state, angles, tesselation_order,
        step_callback=dummy_callback
    )
    
    end_time = time.time()
    print(f"EIGENVALUE-BASED dynamic: {end_time - start_time:.3f} seconds")
    print(f"Final state norm: {np.linalg.norm(final_state):.6f}")
    print(f"Final state sum: {np.sum(np.abs(final_state)**2):.6f}")
    
    return True


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    test_eigenvalue_based_performance()
