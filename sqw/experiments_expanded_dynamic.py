"""
Memory-efficient dynamic quantum walk implementation for large systems.

This module provides a streaming implementation for dynamic quantum walks that:
1. Uses sparse matrices for memory efficiency  
2. Streams computation (doesn't store all states in memory)
3. Applies dynamic angle noise
4. Is a drop-in replacement for the regular running() function
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm as sparse_expm
from itertools import combinations


def sparse_hamiltonian_builder(graph, tesselation, matrix_representation='adjacency'):
    """
    Build Hamiltonian matrix using sparse matrices for memory efficiency.
    This is the sparse equivalent of hamiltonian_builder from experiments_expanded.py
    """
    if matrix_representation == 'adjacency':
        matrix = nx.adjacency_matrix(graph)  # Already sparse
    elif matrix_representation == 'laplacian':
        matrix = nx.laplacian_matrix(graph)  # Already sparse
    else:
        raise ValueError(f"Unknown matrix representation: {matrix_representation}")
        
    num_nodes = graph.number_of_nodes()
    hamiltonian = csr_matrix((num_nodes, num_nodes), dtype=np.float64)
    
    # Convert to lil_matrix for efficient item assignment
    hamiltonian = hamiltonian.tolil()
    matrix = matrix.tolil()
    
    for tesselation_subset in tesselation:
        edge_combinations = combinations(tesselation_subset, 2)
        
        for edge in edge_combinations:
            node_i, node_j = edge[0], edge[1]
            hamiltonian[node_i, node_j] = matrix[node_i, node_j]
            hamiltonian[node_j, node_i] = matrix[node_i, node_j]
            hamiltonian[node_i, node_i] = matrix[node_i, node_i]
            hamiltonian[node_j, node_j] = matrix[node_j, node_j]

    return hamiltonian.tocsr()  # Convert back to csr for efficient operations


def sparse_unitary_builder(hamiltonians, rotation_angles):
    """
    Build unitary operators using sparse matrices for memory efficiency.
    This is the sparse equivalent of unitary_builder from experiments_expanded.py
    """
    num_hamiltonians = len(hamiltonians)
    unitary_operators = []
    
    for hamiltonian_idx in range(num_hamiltonians):
        hamiltonian = hamiltonians[hamiltonian_idx]
        angle = rotation_angles[hamiltonian_idx]
        
        # Use sparse matrix exponential
        evolution_operator = sparse_expm(-1j * angle * hamiltonian)
        unitary_operators.append(evolution_operator)
    
    return unitary_operators


def running_streaming_dynamic_optimized(graph, tesselation_list, num_steps, 
                                      initial_state, angles, tesselation_order,
                                      matrix_representation='adjacency',
                                      searching=[], step_callback=None):
    """
    Memory-efficient dynamic quantum walk that is a drop-in replacement for the
    original running() function but uses streaming computation.
    
    This function:
    1. Uses sparse matrices for all operations
    2. Doesn't store intermediate states (uses callback for streaming)
    3. Has identical interface to the original running() function
    
    Parameters:
    - graph: NetworkX graph
    - tesselation_list: List of tesselations
    - num_steps: Number of evolution steps
    - initial_state: Initial quantum state vector
    - angles: List of angles for each step [step][tesselation_idx]
    - tesselation_order: Order of tesselation application [step][tesselation_idx]
    - matrix_representation: 'adjacency' or 'laplacian'
    - searching: List of search nodes (adds -1 phase)
    - step_callback: Function called for each step with (step_idx, state)
    
    Returns:
    - final_state: Final quantum state
    """
    # Initialize variables
    num_nodes = graph.number_of_nodes()
    num_tesselations = len(tesselation_list)
    
    # Create sparse hamiltonians for every tesselation (only once)
    hamiltonians = []
    for tesselation_idx in range(num_tesselations):
        hamiltonian_matrix = sparse_hamiltonian_builder(
            graph, tesselation_list[tesselation_idx], matrix_representation
        )
        hamiltonians.append(hamiltonian_matrix)

    # Save initial state if callback provided
    if step_callback:
        step_callback(0, initial_state.copy())

    # Evolution loop
    current_state = initial_state.copy()
    
    for time_step in range(num_steps):
        # Build unitary operators for this time step
        rotation_angles = angles[time_step]
        unitary_operators = sparse_unitary_builder(hamiltonians, rotation_angles)

        # Initialize total unitary operator
        if searching == []:
            total_unitary = eye(num_nodes, format='csr', dtype=np.complex128)
        else: 
            total_unitary = eye(num_nodes, format='csr', dtype=np.complex128)
            total_unitary = total_unitary.tolil()  # Convert for efficient item assignment
            for search_node in searching:
                total_unitary[search_node, search_node] = -1
            total_unitary = total_unitary.tocsr()  # Convert back
        
        # Build master unitary operator
        for unitary_idx in range(num_tesselations):
            tesselation_idx = tesselation_order[time_step][unitary_idx]
            total_unitary = unitary_operators[tesselation_idx] @ total_unitary

        # Apply evolution operator to current state
        new_state = total_unitary @ current_state
        
        # Replace old state with new one to minimize memory usage
        current_state = new_state
        del new_state  # Explicit cleanup
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(time_step + 1, current_state.copy())
                   
    return current_state


def estimate_dynamic_memory_usage(N, num_steps):
    """Estimate memory usage for dynamic sparse implementation"""
    # For cycle graphs, each node has 2 neighbors
    nonzeros_per_matrix = 2 * N
    
    # Each sparse matrix element: 8 bytes (float64) + 4 bytes (int32 index) = 12 bytes
    sparse_matrix_size = nonzeros_per_matrix * 12
    
    # State vector: N complex numbers = N * 16 bytes
    state_size = N * 16
    
    # We only need:
    # - 2 Hamiltonians per step (created and destroyed each step)
    # - 1 evolution operator per step (created and destroyed each step)  
    # - 1 current state (persistent)
    # - 1 temporary state during evolution (created and destroyed each step)
    
    # Peak memory per step
    peak_memory_per_step = 3 * sparse_matrix_size + 2 * state_size
    
    # Total persistent memory (only current state persists)
    persistent_memory = state_size
    
    return {
        'sparse_matrix_mb': sparse_matrix_size / (1024**2),
        'state_mb': state_size / (1024**2),
        'peak_per_step_mb': peak_memory_per_step / (1024**2),
        'persistent_mb': persistent_memory / (1024**2),
        'total_for_old_approach_mb': (num_steps * state_size) / (1024**2)  # What the old approach would use
    }


if __name__ == "__main__":
    # Test memory estimation
    print("Memory usage comparison for dynamic quantum walks:")
    print("=" * 60)
    
    for N in [1000, 5000, 10000, 20000]:
        num_steps = N // 4
        mem = estimate_dynamic_memory_usage(N, num_steps)
        print(f"N={N:5d}, steps={num_steps:4d}:")
        print(f"  New approach - Peak per step: {mem['peak_per_step_mb']:6.1f}MB, "
              f"Persistent: {mem['persistent_mb']:5.1f}MB")
        print(f"  Old approach - Total: {mem['total_for_old_approach_mb']:8.1f}MB")
        print(f"  Memory reduction: {mem['total_for_old_approach_mb']/mem['persistent_mb']:6.1f}x")
        print()
