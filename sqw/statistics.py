import numpy as np
from sqw.states import amp2prob

def states2mean(states, domain):
    N = len(domain)
    mean_values = []
    
    for x in range(len(states)):
        mean = 0
        for y in range(N):
            mean += domain[y] * np.real(states[x][y] * states[x][y].conjugate())
            
        mean_values.append(mean)
    
    return mean_values

std
ipr
survival
