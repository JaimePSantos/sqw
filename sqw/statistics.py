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


def states2std(states, domain):
    N = len(domain)
    mean_values = states2mean(states, domain)
    moment_values = []
    
    for x in range(len(states)):
        moment = 0
        for y in range(N):
            moment += (np.real(states[x][y] * states[x][y].conjugate())) * (domain[y] - mean_values[x]) ** 2 
            
        moment_values.append(np.sqrt(moment))
        
    return moment_values

def states2ipr(states, domain):
    N = len(domain)
    ipr_values = []
    for x in range(len(states)):
        ipr = 0
        for y in range(N):
            ipr += (np.real(states[x][y] * states[x][y].conjugate()))**2
            
        ipr_values.append(1/ipr)
        

    return ipr_values

def states2survival(states, node):
    survival_values = []
    
    for x in range(len(states)):
            survival_values.append(np.real(states[x][node] * states[x][node].conjugate()))
    
    return survival_values
