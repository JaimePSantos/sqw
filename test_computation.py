#!/usr/bin/env python3
"""
Test script to verify actual quantum walk computation for different theta values.
This will help us see if the issue is in the computation itself.
"""

import math
import numpy as np
import sys
import os

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_quantum_walk_computation():
    """Test actual quantum walk computation for different theta values"""
    
    from sqw.experiments_expanded_static import create_noise_lists, cycle_tesselation_alpha, cycle_tesselation_beta
    from sqw.experiments_sparse import running_streaming_sparse
    
    # Test parameters
    N = 10  # Small system for testing
    num_steps = 5  # Just a few steps
    initial_nodes = [N//2]  # Start in the middle
    deviation_range = (0, 0)  # No noise case that's causing issues
    
    # Test different theta values
    theta_values = [
        math.pi / 3,  # π/3 ≈ 1.047198
        math.pi,      # π ≈ 3.141593
    ]
    
    print("=== QUANTUM WALK COMPUTATION TEST ===")
    print(f"N = {N}, steps = {num_steps}, deviation = {deviation_range}")
    print()
    
    for theta in theta_values:
        print(f"Testing theta = {theta:.6f} ({theta/math.pi:.3f}π)")
        
        # Test noise list creation
        red_graph = cycle_tesselation_alpha(N)
        blue_graph = cycle_tesselation_beta(N)
        red_edge_list = list(red_graph.edges())
        blue_edge_list = list(blue_graph.edges())
        
        red_noise_list, blue_noise_list = create_noise_lists(
            theta, red_edge_list, blue_edge_list, deviation_range
        )
        
        print(f"  Red noise list (first 5): {red_noise_list[:5]}")
        print(f"  Blue noise list (first 5): {blue_noise_list[:5]}")
        print(f"  Expected: all red = {theta:.6f}, all blue = {theta:.6f}")
        
        # Check if noise lists are as expected for (0,0) case
        expected_red = [theta] * len(red_edge_list)
        expected_blue = [theta] * len(blue_edge_list) 
        
        red_match = np.allclose(red_noise_list, expected_red)
        blue_match = np.allclose(blue_noise_list, expected_blue)
        
        print(f"  Red noise correct: {red_match}")
        print(f"  Blue noise correct: {blue_match}")
        
        # Run a short quantum walk to see the final state
        final_results = []
        
        def step_callback(step, state):
            """Callback to collect states during quantum walk"""
            final_results.append(state.copy())
        
        try:
            running_streaming_sparse(
                N=N,
                theta=theta, 
                num_steps=num_steps,
                initial_nodes=initial_nodes,
                deviation_range=deviation_range,
                step_callback=step_callback
            )
            
            if final_results:
                final_state = final_results[-1]
                # Compute probability distribution
                final_probs = np.abs(final_state)**2
                print(f"  Final probability distribution: {final_probs}")
                print(f"  Max probability at position: {np.argmax(final_probs)}")
                print(f"  Total probability (should be 1): {np.sum(final_probs):.6f}")
            else:
                print("  ERROR: No states collected")
                
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()

if __name__ == "__main__":
    test_quantum_walk_computation()
