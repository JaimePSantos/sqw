import numpy as np

def even_cycle_two_tesselation(N):
    tesselations = []
    for c in range(2):
        tesselation_aux = []
        for x in range(N//2):
            tesselation_aux.append([2*x + c, (2*x + 1 + c) % N])
            
        tesselations.append(tesselation_aux)
       
    return tesselations

    
def square_grid_tesselation(N, periodic = True):
    tesselations = []
    if periodic == False:
        flag = 1
    else:
        flag = 0

    # red
    tesselation_aux = []
    for y in range(N//2):
        for x in range(N):
            tesselation_aux.append([2* y * N + x, (2*y+1)*N + x])

    tesselations.append(tesselation_aux)
    
    # blue
    tesselation_aux = []
    for x in range(N//2):
        for y in range(N):
            tesselation_aux.append([2*x + y*N, 2*x + y*N + 1])
    tesselations.append(tesselation_aux)
    
    # green       
    tesselation_aux = []
    for y in range(N//2 - flag):
        for x in range(N):
            tesselation_aux.append([(2*y+1)*N + x, ((2*y+2)*N + x)%(N**2)])
    tesselations.append(tesselation_aux)
            
    # yellow
    tesselation_aux = []
    for x in range(N//2 - flag):
        for y in range(N):
            if x == N//2-1:
                tesselation_aux.append([2*x + y*N + 1, 2*x + y*N + 2 - N])
            else:
                tesselation_aux.append([2*x + y*N + 1, 2*x + y*N + 2])
    tesselations.append(tesselation_aux)

    
    
    return tesselations