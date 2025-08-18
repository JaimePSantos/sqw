#!/usr/bin/env python3
"""
Test script to verify the new deviation range functionality
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sqw'))

from experiments_expanded_static import create_noise_lists
import numpy as np

def test_deviation_formats():
    """Test different deviation range formats"""
    
    # Mock edge lists for testing
    red_edges = [(0, 1), (2, 3), (4, 5)]
    blue_edges = [(1, 2), (3, 4), (5, 0)]
    theta = np.pi / 4
    
    print("Testing Deviation Range Formats")
    print("=" * 40)
    
    # Test 1: Old format (single value)
    print("\n1. Old format (single value): 0.1")
    red_noise, blue_noise = create_noise_lists(theta, red_edges, blue_edges, 0.1)
    print(f"Red noise deviations: {[f'{x - theta:.4f}' for x in red_noise]}")
    print(f"Blue noise deviations: {[f'{theta - x:.4f}' for x in blue_noise]}")
    
    # Test 2: New format (max_dev, min_factor)
    print("\n2. New format (max_dev=0.2, min_factor=0.3):")
    red_noise, blue_noise = create_noise_lists(theta, red_edges, blue_edges, (0.2, 0.3))
    red_devs = [x - theta for x in red_noise]
    blue_devs = [theta - x for x in blue_noise]
    print(f"Red noise deviations: {[f'{x:.4f}' for x in red_devs]}")
    print(f"Blue noise deviations: {[f'{x:.4f}' for x in blue_devs]}")
    print(f"Red dev range: [{min(red_devs):.4f}, {max(red_devs):.4f}]")
    print(f"Blue dev range: [{min(blue_devs):.4f}, {max(blue_devs):.4f}]")
    print(f"Expected range: [0.060, 0.200]")
    
    # Test 3: Legacy format (explicit min, max)
    print("\n3. Legacy format (explicit min=2, max=5):")
    red_noise, blue_noise = create_noise_lists(theta, red_edges, blue_edges, (2, 5))
    red_devs = [x - theta for x in red_noise]
    blue_devs = [theta - x for x in blue_noise]
    print(f"Red noise deviations: {[f'{x:.4f}' for x in red_devs]}")
    print(f"Blue noise deviations: {[f'{x:.4f}' for x in blue_devs]}")
    print(f"Red dev range: [{min(red_devs):.4f}, {max(red_devs):.4f}]")
    print(f"Blue dev range: [{min(blue_devs):.4f}, {max(blue_devs):.4f}]")
    print(f"Expected range: [2.000, 5.000]")
    
    # Test 4: Multiple runs to show range consistency
    print("\n4. Range consistency test (10 runs with max_dev=0.1, min_factor=0.5):")
    all_red_devs = []
    all_blue_devs = []
    
    for i in range(10):
        red_noise, blue_noise = create_noise_lists(theta, red_edges, blue_edges, (0.1, 0.5))
        red_devs = [x - theta for x in red_noise]
        blue_devs = [theta - x for x in blue_noise]
        all_red_devs.extend(red_devs)
        all_blue_devs.extend(blue_devs)
    
    print(f"Overall red dev range: [{min(all_red_devs):.4f}, {max(all_red_devs):.4f}]")
    print(f"Overall blue dev range: [{min(all_blue_devs):.4f}, {max(all_blue_devs):.4f}]")
    print(f"Expected range: [0.050, 0.100]")

if __name__ == "__main__":
    test_deviation_formats()
