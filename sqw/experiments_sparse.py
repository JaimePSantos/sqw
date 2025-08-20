"""
Memory-efficient sparse matrix implementation for large quantum walks.
This module provides sparse matrix alternatives for the streaming quantum walk computation.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm as sparse_expm
from scipy.sparse import diags


def get_sparse_adjacency_matrix(G):
    """Get adjacency matrix from NetworkX graph as sparse matrix"""
    nodes = sorted(G.nodes())
    return nx.adjacency_matrix(G, nodelist=nodes)  # Returns sparse matrix by default


def create_sparse_noisy_hamiltonians(red_graph, blue_graph, red_noise_list, blue_noise_list):
    """
    Create noisy Hamiltonian matrices using sparse matrices for memory efficiency
    """
    # Get sparse adjacency matrices
    Hr = get_sparse_adjacency_matrix(red_graph).astype(np.float64)
    Hb = get_sparse_adjacency_matrix(blue_graph).astype(np.float64)
    
    # Apply red noise parameters
    red_edge_list = list(red_graph.edges())
    for i, edge in enumerate(red_edge_list):
        if i < len(red_noise_list):
            x, y = edge
            # Apply noise parameter to both symmetric positions
            Hr[x, y] *= red_noise_list[i]
            Hr[y, x] *= red_noise_list[i]
    
    # Apply blue noise parameters
    blue_edge_list = list(blue_graph.edges())
    for i, edge in enumerate(blue_edge_list):
        if i < len(blue_noise_list):
            x, y = edge
            # Apply noise parameter to both symmetric positions
            Hb[x, y] *= blue_noise_list[i]
            Hb[y, x] *= blue_noise_list[i]
    
    return Hr, Hb


def sparse_ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list):
    """Create time evolution operator using sparse matrices"""
    Hr_noisy, Hb_noisy = create_sparse_noisy_hamiltonians(
        red_graph, blue_graph, red_noise_list, blue_noise_list
    )
    
    # Use sparse matrix exponential
    R = sparse_expm(1j * Hr_noisy)
    B = sparse_expm(1j * Hb_noisy)
    
    # Matrix multiplication with sparse matrices
    U = B @ R
    
    return U, Hr_noisy, Hb_noisy


def running_streaming_sparse(N, theta, num_steps, 
                           initial_nodes=[], 
                           deviation_range=0.0,
                           step_callback=None):
    """
    Run staggered quantum walk with static noise using sparse matrices for memory efficiency.
    
    This version uses sparse matrices throughout to minimize memory usage.
    """
    from sqw.experiments_expanded_static import (
        cycle_tesselation_alpha, cycle_tesselation_beta, 
        create_noise_lists, uniform_initial_state
    )
    
    # Create tessellations
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    # Get edge lists
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    
    # Create noise parameters
    red_noise_list, blue_noise_list = create_noise_lists(
        theta, red_edge_list, blue_edge_list, deviation_range
    )
    
    # Create initial state
    psi0 = uniform_initial_state(N, initial_nodes)
    
    # Create evolution operator using sparse matrices
    U, Hr_noisy, Hb_noisy = sparse_ct_evo_with_noise(
        red_graph, blue_graph, red_noise_list, blue_noise_list
    )
    
    # Save initial state if callback provided
    if step_callback:
        step_callback(0, psi0.copy())
    
    # Evolve state step by step, calling callback for each step
    psi = psi0.copy()
    for i in range(num_steps):
        # Apply evolution operator (sparse matrix-vector multiplication)
        psi_new = U @ psi
        
        # Replace old state with new one to minimize memory usage
        psi = psi_new
        del psi_new  # Explicit cleanup
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(i + 1, psi.copy())
    
    return psi


def estimate_sparse_memory_usage(N):
    """Estimate memory usage for sparse matrix implementation"""
    # For cycle graphs, each node has 2 neighbors (except boundary conditions)
    # So adjacency matrix has ~2N non-zero elements
    nonzeros_per_matrix = 2 * N
    
    # Each sparse matrix element: 8 bytes (float64) + 4 bytes (int32 index) = 12 bytes
    sparse_matrix_size = nonzeros_per_matrix * 12
    
    # State vector: N complex numbers = N * 16 bytes
    state_size = N * 16
    
    # Total memory (2 Hamiltonians + 1 evolution operator + 1 state)
    total_memory = 3 * sparse_matrix_size + state_size
    
    return {
        'sparse_matrix_mb': sparse_matrix_size / (1024**2),
        'state_mb': state_size / (1024**2),
        'total_mb': total_memory / (1024**2)
    }


if __name__ == "__main__":
    # Test memory estimation
    for N in [1000, 5000, 10000, 20000]:
        mem = estimate_sparse_memory_usage(N)
        print(f"N={N:5d}: Sparse matrix={mem['sparse_matrix_mb']:6.1f}MB, "
              f"State={mem['state_mb']:5.1f}MB, Total={mem['total_mb']:7.1f}MB")
