#!/usr/bin/env python3

"""
Test script for unified folder structure
"""

import sys
import os
sys.path.append('.')

from smart_loading_static import get_experiment_dir
from static_local_logged_mp import dummy_tesselation_func
import math

print("Testing unified folder structure:")

# Test with dev=0 (no noise)
dev_0_path = get_experiment_dir(
    dummy_tesselation_func, 
    False, 
    106, 
    noise_params=[0], 
    noise_type='static_noise', 
    base_dir='test_experiments', 
    theta=math.pi/2
)
print(f"Dev 0: {dev_0_path}")

# Test with dev=0.01 (with noise)
dev_001_path = get_experiment_dir(
    dummy_tesselation_func, 
    True, 
    106, 
    noise_params=[0.01], 
    noise_type='static_noise', 
    base_dir='test_experiments', 
    theta=math.pi/2
)
print(f"Dev 0.01: {dev_001_path}")

# Check if paths have unified structure
print("\nStructure analysis:")
print(f"Both paths use same base folder: {'dummy_tesselation_func_static_noise' in dev_0_path and 'dummy_tesselation_func_static_noise' in dev_001_path}")
print(f"No noise/no-noise separation: {'static_noise_nonoise' not in dev_0_path and 'static_noise_nonoise' not in dev_001_path}")
print(f"Dev 0 includes dev folder: {'dev_0.000' in dev_0_path}")
print(f"Dev 0.01 includes dev folder: {'dev_0.010' in dev_001_path}")
