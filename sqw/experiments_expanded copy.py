from scipy.linalg import expm
import numpy as np
from itertools import combinations
import networkx as nx

def hamiltonian_builder(graph, tesselation, matrix_representation):
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

def unitary_builder(hamiltonians, rotation_angles):
    num_hamiltonians = len(hamiltonians)
    unitary_operators = []
    
    for hamiltonian_idx in range(num_hamiltonians):
        eigenvalues_matrix, eigenvectors_matrix = hamiltonians[hamiltonian_idx]
        evolution_operator = expm(-1j * rotation_angles[hamiltonian_idx] * eigenvalues_matrix)
        unitary_op = eigenvectors_matrix @ evolution_operator @ eigenvectors_matrix.H
        unitary_operators.append(unitary_op)
    
    return unitary_operators

def running(graph, tesselation_list, num_steps, 
            initial_state, 
            angles=[], 
            tesselation_order=[],
            matrix_representation='adjacency',
            searching=[]):
    # Initialize variables
    evolution_states = [initial_state]
    hamiltonians = []
    
    num_nodes = graph.number_of_nodes()
    num_tesselations = len(tesselation_list)
    
    # Create hamiltonians for every tesselation
    for tesselation_idx in range(num_tesselations):
        hamiltonian_matrix = hamiltonian_builder(graph, tesselation_list[tesselation_idx], matrix_representation)
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
        hamiltonians.append((np.diag(eigenvalues), np.matrix(eigenvectors)))

    # Evolution loop
    current_state = initial_state
    for time_step in range(num_steps):
        unitary_operators = unitary_builder(hamiltonians, angles[time_step])

        if searching == []:
            total_unitary = np.eye(num_nodes, dtype='complex')
        else: 
            total_unitary = np.eye(num_nodes, dtype='complex')
            for search_node in searching:
                total_unitary[search_node, search_node] = -1
        
        # Build master unitary operator
        for unitary_idx in range(num_tesselations):
            tesselation_idx = tesselation_order[time_step][unitary_idx]
            total_unitary = unitary_operators[tesselation_idx] @ total_unitary

        current_state = total_unitary @ current_state
        evolution_states.append(np.array(current_state))
                   
    return evolution_states