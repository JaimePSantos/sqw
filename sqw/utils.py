"""
Utility Functions for Quantum Walk Experiments

This module provides utility functions for random angle generation,
tesselation selection, and other supporting operations for
quantum walk simulations.
"""

import numpy as np
import random

def random_angle_deviation(angle_values, angle_dev, nof_steps):
    """
    Generate random angle deviations for dynamic quantum walks.
    
    Args:
        angle_values: Base angle values
        angle_dev: Deviation values for each angle
        nof_steps: Number of time steps
    
    Returns:
        List of angle lists with random deviations
    """
    nof_angles = len(angle_values)
    angles_list = []
    
    for x in range(nof_steps):
        angles_list.append([angle_values[a] + random.uniform(-angle_dev[a],angle_dev[a]) for a in range(nof_angles)])

    return angles_list

def tesselation_choice(tesselations_list, steps, prob):
    """
    Choose tesselation orders based on probability distribution.
    
    Args:
        tesselations_list: List of available tesselations
        steps: Number of time steps  
        prob: Probability distribution for choosing tesselations
    
    Returns:
        List of chosen tesselation orders
    """
    nof_tesselations = len(tesselations_list)
    tesselation_final = []
    
    for x in range(steps):
        index = np.random.choice(range(nof_tesselations), p = prob)
        tesselation_final.append(tesselations_list[index])

    return tesselation_final
