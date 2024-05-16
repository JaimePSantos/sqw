from scipy.linalg import expm
import numpy as np
from itertools import combinations
import networkx as nx

def hamiltonian_builder(G, T, matrix_representation):
    if matrix_representation == 'adjacency':
        M = nx.adjacency_matrix(G).todense()
    if matrix_representation == 'laplacian':
        M = nx.laplacian_matrix(G).todense()
        
    N = G.number_of_nodes()
    H = np.zeros((N,N))
    
    for t in T:
        
        combinations_tesselation = combinations(t, 2)
        
        for c in combinations_tesselation:
            H[c[0], c[1]] = M[c[0], c[1]]
            H[c[1], c[0]] = M[c[0], c[1]]
            H[c[0], c[0]] = M[c[0], c[0]]
            H[c[1], c[1]] = M[c[1], c[1]]

    return H

def unitary_builder(H, theta):
    number_of_hamiltonians = len(H)
    U = []
    for x in range(number_of_hamiltonians):
        U.append(  H[x][1] @ (expm(-1j * theta[x] * H[x][0])) @ H[x][1].H  )
    
    return U

def running(G, T, steps, 
            initial_state, 
            angles = [], 
            tesselation_order = [],
            matrix_representation = 'adjacency',
            searching = []):
    # Hamiltonians creations
    state = [] 
    H = []
    final_states = [initial_state]
    
    N = G.number_of_nodes()
    number_of_tesselations = len(T)
    
    # creates the hamiltonians for every tesselation
    for x in range(number_of_tesselations):
        H_aux = hamiltonian_builder(G, T[x], matrix_representation)

        D, V = np.linalg.eigh(H_aux)
        H.append((np.diag(D), np.matrix(V)))

        
    # creates de unitary per steps, and depends on (angles, tesselation order)
    for t in range(steps):
        U = unitary_builder(H, angles[t])

        if searching == []:
            unitary_operator = np.eye(N, dtype = 'complex')
        else: 
            unitary_operator = np.eye(N, dtype = 'complex')
            for u in searching:
                unitary_operator[u,u] = -1
        
        
        # master unitary creation
        for u in range(number_of_tesselations):
            unitary_operator = U[tesselation_order[t][u]] @ unitary_operator

        initial_state = unitary_operator @ initial_state
            
        final_states.append(np.array(initial_state))
                   
    return final_states