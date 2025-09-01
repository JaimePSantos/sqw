"""
SQW Package - Staggered Quantum Walk

This package contains modules for quantum walk simulations and analysis.
Optimized for high-performance dynamic quantum walk experiments.
"""

__version__ = "1.0.0"
__author__ = "Jaime Santos"

# Import key functions for easy access
from .tesselations import even_line_two_tesselation, even_cycle_two_tesselation, square_grid_tesselation
from .states import uniform_initial_state, amp2prob
from .utils import random_angle_deviation, tesselation_choice

__all__ = [
    'even_line_two_tesselation',
    'even_cycle_two_tesselation', 
    'square_grid_tesselation',
    'uniform_initial_state',
    'amp2prob',
    'random_angle_deviation',
    'tesselation_choice'
]
