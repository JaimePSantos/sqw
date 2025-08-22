#!/usr/bin/env python3
"""
Diagnostic test for the (0,0) deviation spreading issue.
This will help identify why the no-noise case is not spreading properly.
"""

import math
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_no_noise_spreading():
    """Test the spreading behavior for different deviation cases"""
    
    from sqw.experiments_sparse import running_streaming_sparse
    
    # Test parameters - smaller system for quick testing
    N = 100
    num_steps = 50  # Fewer steps for quick test
    initial_nodes = [N//2]  # Start in the middle
    theta = math.pi  # Current theta value from your script
    
    # Test different deviation cases
    test_cases = [
        (0, 0),      # No noise - should spread FASTEST
        (0, 0.02),   # Small noise
        (0, 0.1),    # Medium noise  
        (0, 0.5),    # Large noise
    ]
    
    print("=== NO-NOISE SPREADING DIAGNOSTIC TEST ===")
    print(f"N = {N}, steps = {num_steps}, theta = {theta:.6f}")
    print()
    
    results = {}
    
    for deviation_range in test_cases:
        print(f"Testing deviation {deviation_range}...")
        
        # Collect all states during the walk
        all_states = []
        
        def step_callback(step, state):
            """Callback to collect states during quantum walk"""
            all_states.append(state.copy())
        
        try:
            running_streaming_sparse(
                N=N,
                theta=theta, 
                num_steps=num_steps,
                initial_nodes=initial_nodes,
                deviation_range=deviation_range,
                step_callback=step_callback
            )
            
            if all_states:
                # Calculate standard deviation at each time step
                std_values = []
                for state in all_states:
                    prob_dist = np.abs(state.flatten())**2
                    # Calculate position expectation and standard deviation
                    positions = np.arange(len(prob_dist))
                    mean_pos = np.sum(positions * prob_dist)
                    variance = np.sum((positions - mean_pos)**2 * prob_dist)
                    std_dev = np.sqrt(variance)
                    std_values.append(std_dev)
                
                results[deviation_range] = std_values
                
                print(f"  Initial std: {std_values[0]:.4f}")
                print(f"  Final std: {std_values[-1]:.4f}")
                print(f"  Growth factor: {std_values[-1]/std_values[0]:.2f}x")
                print(f"  Average growth rate: {std_values[-1]/num_steps:.4f} per step")
                
            else:
                print(f"  ERROR: No states collected")
                results[deviation_range] = None
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results[deviation_range] = None
        
        print()
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'orange', 'green', 'red']
    for i, (deviation_range, std_values) in enumerate(results.items()):
        if std_values is not None:
            time_steps = range(len(std_values))
            label = f"deviation = {deviation_range}"
            plt.plot(time_steps, std_values, color=colors[i], label=label, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Time - Diagnostic Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig('diagnostic_spreading_test.png', dpi=150, bbox_inches='tight')
    print("Diagnostic plot saved as 'diagnostic_spreading_test.png'")
    
    # Analysis
    print("=== ANALYSIS ===")
    if (0,0) in results and results[(0,0)] is not None:
        no_noise_final = results[(0,0)][-1]
        print(f"No-noise final std: {no_noise_final:.4f}")
        
        # Compare with other cases
        for deviation_range, std_values in results.items():
            if deviation_range != (0,0) and std_values is not None:
                other_final = std_values[-1]
                if other_final > no_noise_final:
                    print(f"ERROR: {deviation_range} spreads MORE than no-noise case!")
                    print(f"  No-noise: {no_noise_final:.4f}")
                    print(f"  {deviation_range}: {other_final:.4f}")
                    print(f"  This should not happen!")
    
    return results

if __name__ == "__main__":
    test_no_noise_spreading()
