#!/usr/bin/env python3

"""
Test script for backward compatibility
"""

import sys
import os
sys.path.append('.')

from smart_loading_static import find_experiment_dir_flexible
from static_local_logged_mp import dummy_tesselation_func
import math

print("Testing backward compatibility:")

# Test finding non-existent directory (should return unified path)
unified_path, format_type = find_experiment_dir_flexible(
    dummy_tesselation_func, 
    True, 
    106, 
    noise_params=[0.01], 
    noise_type='static_noise', 
    base_dir='test_experiments', 
    theta=math.pi/2
)
print(f"Non-existent path: {unified_path} (format: {format_type})")

# Test with dev=0 (no noise)
no_noise_path, format_type = find_experiment_dir_flexible(
    dummy_tesselation_func, 
    False, 
    106, 
    noise_params=[0], 
    noise_type='static_noise', 
    base_dir='test_experiments', 
    theta=math.pi/2
)
print(f"No noise path: {no_noise_path} (format: {format_type})")

print("\nNew unified structure examples:")
print("  With noise: dummy_tesselation_func_static_noise/theta_1.570796/dev_0.010/N_106")
print("  No noise:   dummy_tesselation_func_static_noise/theta_1.570796/dev_0.000/N_106")
print("\nOld separated structure would have been:")
print("  With noise: dummy_tesselation_func_static_noise/theta_1.570796/dev_0.010/N_106")
print("  No noise:   dummy_tesselation_func_static_noise_nonoise/theta_1.570796/N_106")
