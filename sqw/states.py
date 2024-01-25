import numpy as np

def uniform_initial_state(N, nodes = []):
    if nodes == []:
        state = np.ones((N,1)) / np.sqrt(N)
    else:
        state = np.zeros((N,1))
        for x in range(len(nodes)):
            state[nodes[x]] = 1
        state = state / np.sqrt(len(nodes))
    
    return state

def amp2prob(state):
    
    return np.real(state * state.conjugate())