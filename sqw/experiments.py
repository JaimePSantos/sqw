from scipy.linalg import expm
import numpy as np
from itertools import combinations

def hamiltonian_builder(G, T):
    N = G.number_of_nodes()
    H = np.zeros((N,N))
    
    for t in T:
        
        combinations_tesselation = combinations(t, 2)
        
        for c in combinations_tesselation:
            H[c[0], c[1]] = 1
            H[c[1], c[0]] = 1
    
    return H

def unitary_builder(H,theta):
    number_of_hamiltonians = len(H)
    U = []
    for x in range(number_of_hamiltonians):
        U.append(expm(-1j * theta[x] * H[x]))
    
    return U

def running(G, T, steps, initial_state, angles = [], tesselation_order = []):
    # Hamiltonians creations
    state = [] 
    H = []
    final_states = [initial_state]
    
    N = G.number_of_nodes()
    number_of_tesselations = len(T)
    
    # creates the hamiltonians for every tesselation
    for x in range(number_of_tesselations):
        H.append(hamiltonian_builder(G, T[x]))
        
    # creates de unitary per steps, and depends on (angles, tesselation order)
    for t in range(steps):
        U = unitary_builder(H, angles[t])
        
        unitary_operator = np.eye(N, dtype = 'complex')
        
        # master unitary creation
        for u in range(number_of_tesselations):
            unitary_operator = U[tesselation_order[t][u]] @ unitary_operator

        initial_state = unitary_operator @ initial_state
            
        final_states.append(initial_state)
                   
    return final_states