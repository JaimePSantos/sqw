#!/usr/bin/env python3

"""
Performance Comparison Test: Eigenvalue vs Non-Eigenvalue Dynamic Implementations

This script compares the performance of:
1. Original dynamic implementation (using running from experiments_expanded)
2. Eigenvalue-based optimized implementation (using running_streaming_dynamic_eigenvalue_based)

The test will help identify why the current optimized version feels slow.
"""

import time
import math
import numpy as np
import sys
import os

# Add the current directory to the path to import sqw modules
sys.path.insert(0, os.path.dirname(__file__))

def setup_test_parameters():
    """Setup consistent test parameters for both implementations"""
    # Small test parameters for quick comparison
    N = 100
    steps = 25
    base_theta = math.pi/3
    dev = 0.2  # Test with some noise
    
    print(f"Test Parameters:")
    print(f"  N = {N}")
    print(f"  steps = {steps}")
    print(f"  base_theta = {base_theta}")
    print(f"  deviation = {dev}")
    print()
    
    return N, steps, base_theta, dev

def setup_common_environment(N, base_theta):
    """Setup the common environment needed by both implementations"""
    import networkx as nx
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    
    # Create graph and tesselation
    graph = nx.path_graph(N)
    tesselation = even_line_two_tesselation(N)
    
    # Create initial state
    initial_state = uniform_initial_state(N, nodes=[N//2])
    
    print(f"Environment setup:")
    print(f"  Graph: {N} nodes (path graph)")
    print(f"  Tesselation: {len(tesselation)} tesselations")
    print(f"  Initial state: centered at node {N//2}")
    print()
    
    return graph, tesselation, initial_state

def test_original_implementation(graph, tesselation, initial_state, steps, base_theta, dev):
    """Test the original non-eigenvalue implementation"""
    print("=" * 60)
    print("TESTING ORIGINAL IMPLEMENTATION (non-eigenvalue)")
    print("=" * 60)
    
    try:
        from sqw.experiments_expanded import running
        from sqw.utils import random_angle_deviation
        
        # Generate angles for this test - function takes arrays of base values and deviations
        angles = random_angle_deviation([base_theta, base_theta], [dev, dev], steps)
        tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
        
        print("Starting original implementation test...")
        start_time = time.time()
        
        final_state = running(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[]
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Original implementation completed in {duration:.4f} seconds")
        print(f"Final state shape: {final_state.shape}")
        print(f"Final state norm: {np.linalg.norm(final_state):.6f}")
        print()
        
        return duration, final_state
        
    except Exception as e:
        print(f"ERROR in original implementation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_eigenvalue_implementation(graph, tesselation, initial_state, steps, base_theta, dev):
    """Test the eigenvalue-based optimized implementation"""
    print("=" * 60)
    print("TESTING EIGENVALUE IMPLEMENTATION (optimized)")
    print("=" * 60)
    
    try:
        from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
        from sqw.utils import random_angle_deviation
        
        # Generate angles for this test - function takes arrays of base values and deviations
        angles = random_angle_deviation([base_theta, base_theta], [dev, dev], steps)
        tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
        
        print("Starting eigenvalue-based implementation test...")
        start_time = time.time()
        
        final_state = running_streaming_dynamic_eigenvalue_based(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[]
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Eigenvalue implementation completed in {duration:.4f} seconds")
        print(f"Final state shape: {final_state.shape}")
        print(f"Final state norm: {np.linalg.norm(final_state):.6f}")
        print()
        
        return duration, final_state
        
    except Exception as e:
        print(f"ERROR in eigenvalue implementation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def compare_results(original_state, eigenvalue_state, original_time, eigenvalue_time):
    """Compare the results from both implementations"""
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    if original_time is not None and eigenvalue_time is not None:
        speedup = original_time / eigenvalue_time
        print(f"Performance Comparison:")
        print(f"  Original time:    {original_time:.4f} seconds")
        print(f"  Eigenvalue time:  {eigenvalue_time:.4f} seconds")
        print(f"  Speedup factor:   {speedup:.2f}x")
        
        if speedup > 1:
            print(f"  ✓ Eigenvalue implementation is {speedup:.2f}x FASTER")
        else:
            print(f"  ✗ Eigenvalue implementation is {1/speedup:.2f}x SLOWER")
        print()
    
    if original_state is not None and eigenvalue_state is not None:
        # Compare final states
        state_diff = np.linalg.norm(original_state - eigenvalue_state)
        print(f"State Comparison:")
        print(f"  State difference (norm): {state_diff:.8f}")
        
        if state_diff < 1e-10:
            print(f"  ✓ States are IDENTICAL (within numerical precision)")
        elif state_diff < 1e-6:
            print(f"  ✓ States are VERY CLOSE (acceptable difference)")
        else:
            print(f"  ✗ States are DIFFERENT (possible implementation error)")
        print()

def main():
    """Main test function"""
    print("Dynamic Quantum Walk Implementation Performance Test")
    print("=" * 60)
    print()
    
    # Setup test parameters
    N, steps, base_theta, dev = setup_test_parameters()
    
    # Setup common environment
    graph, tesselation, initial_state = setup_common_environment(N, base_theta)
    
    # Test original implementation
    original_time, original_state = test_original_implementation(
        graph, tesselation, initial_state, steps, base_theta, dev
    )
    
    # Test eigenvalue implementation
    eigenvalue_time, eigenvalue_state = test_eigenvalue_implementation(
        graph, tesselation, initial_state, steps, base_theta, dev
    )
    
    # Compare results
    compare_results(original_state, eigenvalue_state, original_time, eigenvalue_time)
    
    # Additional analysis
    if original_time is not None and eigenvalue_time is not None:
        print("Analysis:")
        if eigenvalue_time > original_time:
            slowdown = eigenvalue_time / original_time
            print(f"  The eigenvalue implementation is {slowdown:.2f}x slower than expected!")
            print(f"  This suggests the optimization is not working as intended.")
            print(f"  Possible issues:")
            print(f"    - Eigenvalue decomposition overhead")
            print(f"    - Memory allocation issues")
            print(f"    - Matrix reconstruction overhead")
            print(f"    - Non-optimized matrix operations")
        else:
            print(f"  The eigenvalue implementation is working as expected.")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
