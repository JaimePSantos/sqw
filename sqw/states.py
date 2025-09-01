"""
State Functions for Quantum Walk Experiments

This module provides functions for creating and manipulating
quantum walk states, including initial state preparation and
probability calculations.
"""

import numpy as np

def uniform_initial_state(N, nodes=[]):
    """
    Create a uniform initial state for quantum walk.
    If nodes is empty, creates equal superposition of all N nodes.
    If nodes is provided, creates equal superposition of specified nodes.
    """
    initial_state = np.zeros(N, dtype=np.complex128)
    
    if not nodes:
        # Equal superposition of all nodes
        initial_state[:] = 1.0 / np.sqrt(N)
    else:
        # Equal superposition of specified nodes
        initial_state[nodes] = 1.0 / np.sqrt(len(nodes))
    
    return initial_state

def amp2prob(amplitude_array):
    """
    Convert amplitude array to probability array.
    
    Args:
        amplitude_array: Complex amplitude array
        
    Returns:
        Real probability array (|amplitude|^2)
    """
    return np.abs(amplitude_array) ** 2
