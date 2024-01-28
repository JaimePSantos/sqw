import numpy as np
import random

def random_tesselation_order(nof_tesselations, nof_steps, tess_prob):
    
    tesselation_list = []
    
    for x in range(nof_steps):
        tesselation_list.append([np.random.choice(np.arange(0, nof_tesselations), p = tess_prob) for x in range(nof_tesselations)])
    
    return tesselation_list

def random_angle_deviation(angle_values, angle_dev, nof_steps):
    
    nof_angles = len(angle_values)
    angles_list = []
    
    for x in range(nof_steps):
        angles_list.append([angle_values[a] + random.uniform(-angle_dev[a],angle_dev[a]) for a in range(nof_angles)])

    return angles_list


def tesselation_choice(tesselations_list, steps, prob):
    
    nof_tesselations = len(tesselations_list)
    tesselation_final = []
    
    for x in range(steps):
        index = np.random.choice(range(nof_tesselations), p = prob)
        tesselation_final.append(tesselations_list[index])

    return tesselation_final
