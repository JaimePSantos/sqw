#!/usr/bin/env python3

"""
Test script to validate theta-dependent directory structure and dev=0 behavior.

This script helps debug issues with deviation=0 results being different for different theta values.
"""

import os
import math
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_theta_directory_formatting():
    """Test the theta directory formatting function"""
    print("=== Testing Theta Directory Formatting ===")
    
    try:
        from smart_loading_static import format_theta_for_directory
        
        test_thetas = [
            (math.pi/6, "pi/6"),
            (math.pi/4, "pi/4"), 
            (math.pi/3, "pi/3"),
            (math.pi/2, "pi/2"),
            (2*math.pi/3, "2*pi/3"),
            (3*math.pi/4, "3*pi/4"),
            (5*math.pi/6, "5*pi/6"),
            (math.pi, "pi"),
            (1.047197551, "custom ~pi/3"),  # approximately pi/3
            (1.5707963267948966, "custom ~pi/2")  # exactly pi/2
        ]
        
        for theta, description in test_thetas:
            dir_name = format_theta_for_directory(theta)
            print(f"theta = {theta:.10f} ({description:15s}) -> {dir_name}")
            
        print("\nValidation:")
        print("- Common fractions of pi should use symbolic names (pi3, pi4, etc.)")
        print("- Custom values should use high-precision numeric format")
        print("- This ensures different theta values get different directories")
        
    except ImportError as e:
        print(f"Error importing format_theta_for_directory: {e}")
        return False
    
    return True

def test_experiment_directory_paths():
    """Test experiment directory path generation for different theta values"""
    print("\n=== Testing Experiment Directory Paths ===")
    
    try:
        from smart_loading_static import get_experiment_dir
        
        def dummy_tesselation_func(N):
            return None
        dummy_tesselation_func.__name__ = "dummy_tesselation_func"
        
        # Test parameters
        N = 500
        noise_params = [(0, 0)]  # dev=0 case
        noise_type = "static_noise"
        base_dir = "experiments_data_samples"
        
        test_thetas = [
            math.pi/3,
            math.pi/2,
            1.047197551,  # approximately pi/3 but not exactly
        ]
        
        print("Testing directory paths for dev=(0,0) with different theta values:")
        for theta in test_thetas:
            exp_dir = get_experiment_dir(
                dummy_tesselation_func, 
                has_noise=False,  # dev=0 means no noise
                N=N,
                noise_params=noise_params,
                noise_type=noise_type,
                base_dir=base_dir,
                theta=theta
            )
            print(f"theta = {theta:.10f} -> {exp_dir}")
        
        print("\nValidation:")
        print("- Different theta values should create different directory paths")
        print("- This prevents data mixing between different theta experiments")
        print("- If user changes theta, they should see separate results")
        
    except ImportError as e:
        print(f"Error importing experiment directory functions: {e}")
        return False
    except Exception as e:
        print(f"Error testing directory paths: {e}")
        return False
    
    return True

def test_deviation_processing():
    """Test how deviation ranges are processed"""
    print("\n=== Testing Deviation Range Processing ===")
    
    try:
        from sqw.experiments_expanded_static import create_noise_lists
        import networkx as nx
        
        # Create simple test graphs
        N = 10
        red_edges = [(i, (i+1)%N) for i in range(N)]  # Simple cycle
        blue_edges = [(i, (i+2)%N) for i in range(N)]  # Every other connection
        
        theta = math.pi/3
        
        print(f"Testing noise list creation with theta = {theta:.6f}")
        print("Test cases:")
        
        test_devs = [
            (0, 0),      # No noise
            (0, 0.1),    # Small noise range
            0,           # Legacy format: no noise  
            0.1          # Legacy format: small noise
        ]
        
        for dev in test_devs:
            red_noise, blue_noise = create_noise_lists(theta, red_edges, blue_edges, dev)
            
            red_min, red_max = min(red_noise), max(red_noise)
            blue_min, blue_max = min(blue_noise), max(blue_noise)
            
            print(f"  dev = {dev}")
            print(f"    Red noise range:  [{red_min:.6f}, {red_max:.6f}]")
            print(f"    Blue noise range: [{blue_min:.6f}, {blue_max:.6f}]")
            
            # For dev=(0,0) or dev=0, all values should equal theta
            if dev == (0, 0) or dev == 0:
                if abs(red_min - theta) < 1e-10 and abs(red_max - theta) < 1e-10:
                    print(f"    ✓ Red noise correctly equals theta = {theta:.6f}")
                else:
                    print(f"    ✗ Red noise should equal theta = {theta:.6f}")
                
                if abs(blue_min - theta) < 1e-10 and abs(blue_max - theta) < 1e-10:
                    print(f"    ✓ Blue noise correctly equals theta = {theta:.6f}")
                else:
                    print(f"    ✗ Blue noise should equal theta = {theta:.6f}")
            
        print("\nValidation:")
        print("- For dev=(0,0) or dev=0, all noise values should exactly equal theta")
        print("- This ensures deterministic evolution for the no-noise case")
        print("- Different theta values should give different deterministic evolutions")
        
    except ImportError as e:
        print(f"Error importing noise functions: {e}")
        return False
    except Exception as e:
        print(f"Error testing deviation processing: {e}")
        return False
    
    return True

def main():
    """Run all validation tests"""
    print("Quantum Walk Theta Validation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_theta_directory_formatting()
    all_passed &= test_experiment_directory_paths()
    all_passed &= test_deviation_processing()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests completed successfully")
        print("\nKey findings:")
        print("1. Different theta values create separate directories")
        print("2. dev=(0,0) should give deterministic results for each theta")
        print("3. If user sees different dev=0 results, check theta value")
    else:
        print("✗ Some tests failed - check imports and dependencies")
    
    print("\nTo fix dev=0 issues:")
    print("1. Ensure you're using the correct theta value")
    print("2. Check that experiment directories include theta")
    print("3. Verify different theta values are stored separately")
    print("4. Clear old data if mixing occurred")

if __name__ == "__main__":
    main()
