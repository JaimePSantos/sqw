import numpy as np

def even_cycle_two_tesselation(N):
    tesselations = []
    for c in range(2):
        tesselation_aux = []
        for x in range(N//2):
            tesselation_aux.append([2*x + c, (2*x + 1 + c) % N])
            
        tesselations.append(tesselation_aux)
       
    return tesselations

    
    