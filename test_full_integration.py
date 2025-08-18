#!/usr/bin/env python3
"""
Test script to verify the new deviation range functionality in the full system
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'sqw'))

# Test the new formatting function
from smart_loading_static import format_deviation_for_filename, get_experiment_dir

def test_new_format():
    """Test the new deviation range format"""
    
    print("Testing new deviation range format integration")
    print("=" * 50)
    
    # Test different deviation formats
    test_devs = [
        0,              # No noise (single value format)
        (0.1, 0.0),     # Range [0.0, 0.1] - equivalent to old 0.1
        (0.5, 0.2),     # Range [0.1, 0.5] - new format with min factor 0.2
        (1.0, 0.3),     # Range [0.3, 1.0] - new format with min factor 0.3
        (10.0, 0.1)     # Range [1.0, 10.0] - new format with min factor 0.1
    ]
    
    # Test filename formatting
    print("\n1. Testing filename formatting:")
    for dev in test_devs:
        filename_str = format_deviation_for_filename(dev)
        print(f"  {dev} -> {filename_str}")
    
    # Test directory creation
    print("\n2. Testing directory path generation:")
    def dummy_tesselation_func(N):
        return None
    dummy_tesselation_func.__name__ = "dummy_tesselation"
    
    N = 100
    for dev in test_devs:
        # Handle new deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # New format: (max_dev, min_factor) or legacy (min, max)
            if dev[1] <= 1.0 and dev[1] >= 0.0:
                # New format
                max_dev, min_factor = dev
                has_noise = max_dev > 0
            else:
                # Legacy format
                min_val, max_val = dev
                has_noise = max_val > 0
        else:
            # Single value format
            has_noise = dev > 0
        
        noise_params = [dev] if has_noise else [0]
        exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N,
            noise_params=noise_params, 
            noise_type="static_noise",
            base_dir="test_experiments_data"
        )
        print(f"  {dev} -> {exp_dir}")
    
    # Test quantum walk execution with new format
    print("\n3. Testing quantum walk execution:")
    try:
        from sqw.experiments_expanded_static import running
        
        N = 10
        theta = 3.14159/4
        steps = 5
        initial_nodes = [N//2]
        
        for dev in test_devs[:3]:  # Test first 3 for speed
            print(f"\n  Testing dev={dev}:")
            probabilities = running(
                N, theta, steps,
                initial_nodes=initial_nodes,
                deviation_range=dev,
                return_all_states=False
            )
            print(f"    Success! Final probabilities sum: {probabilities.sum():.6f}")
            
    except Exception as e:
        print(f"    Error in quantum walk execution: {e}")
    
    print(f"\n4. Testing edge cases:")
    edge_cases = [
        (0.0, 0.5),     # Zero max deviation
        (1.0, 0.0),     # Zero min factor
        (1.0, 1.0),     # Min factor = 1
        (2.0, 3.0),     # Legacy format (should be treated as min=2, max=3)
    ]
    
    for dev in edge_cases:
        try:
            filename_str = format_deviation_for_filename(dev)
            print(f"  {dev} -> {filename_str}")
        except Exception as e:
            print(f"  {dev} -> ERROR: {e}")

if __name__ == "__main__":
    test_new_format()
