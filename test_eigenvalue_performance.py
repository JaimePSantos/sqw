#!/usr/bin/env python3

"""
Simple Performance Test for Dynamic Quantum Walk Eigenvalue Implementation

This test focuses specifically on the eigenvalue-based implementation
to identify performance bottlenecks.
"""

import time
import math
import numpy as np
import sys
import os

# Add the current directory to the path to import sqw modules
sys.path.insert(0, os.path.dirname(__file__))

def test_eigenvalue_performance():
    """Test the eigenvalue implementation performance with profiling"""
    print("=" * 60)
    print("EIGENVALUE IMPLEMENTATION PERFORMANCE TEST")
    print("=" * 60)
    
    # Test parameters
    N = 100
    steps = 25
    base_theta = math.pi/3
    dev = 0.2
    
    print(f"Parameters: N={N}, steps={steps}, deviation={dev}")
    print()
    
    try:
        # Setup
        import networkx as nx
        from sqw.tesselations import even_line_two_tesselation
        from sqw.states import uniform_initial_state
        from sqw.utils import random_angle_deviation
        from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
        
        # Create environment
        setup_start = time.time()
        graph = nx.path_graph(N)
        tesselation = even_line_two_tesselation(N)
        initial_state = uniform_initial_state(N, nodes=[N//2])
        angles = random_angle_deviation([base_theta, base_theta], [dev, dev], steps)
        tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
        setup_end = time.time()
        
        print(f"Setup time: {setup_end - setup_start:.4f} seconds")
        
        # Run the eigenvalue implementation with detailed timing
        print("Starting eigenvalue implementation...")
        total_start = time.time()
        
        final_state = running_streaming_dynamic_eigenvalue_based(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[]
        )
        
        total_end = time.time()
        total_time = total_end - total_start
        
        print(f"Total execution time: {total_time:.4f} seconds")
        print(f"Time per step: {total_time/steps:.4f} seconds")
        print(f"Final state shape: {final_state.shape}")
        print(f"Final state norm: {np.linalg.norm(final_state):.6f}")
        print()
        
        # Performance analysis
        expected_time_per_step = 0.01  # Conservative estimate for N=100
        if total_time/steps > expected_time_per_step:
            print("⚠️  PERFORMANCE ISSUE DETECTED:")
            print(f"   Current: {total_time/steps:.4f} s/step")
            print(f"   Expected: ~{expected_time_per_step:.4f} s/step")
            print(f"   Slowdown: {(total_time/steps)/expected_time_per_step:.1f}x slower than expected")
        else:
            print("✅ Performance looks good!")
        
        return total_time
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_scaling_performance():
    """Test how performance scales with problem size"""
    print("=" * 60)
    print("SCALING PERFORMANCE TEST")
    print("=" * 60)
    
    sizes = [50, 100, 200]
    steps_list = [10, 25, 50]
    
    for N, steps in zip(sizes, steps_list):
        print(f"\nTesting N={N}, steps={steps}:")
        
        try:
            import networkx as nx
            from sqw.tesselations import even_line_two_tesselation
            from sqw.states import uniform_initial_state
            from sqw.utils import random_angle_deviation
            from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
            
            # Setup
            graph = nx.path_graph(N)
            tesselation = even_line_two_tesselation(N)
            initial_state = uniform_initial_state(N, nodes=[N//2])
            angles = random_angle_deviation([math.pi/3, math.pi/3], [0.2, 0.2], steps)
            tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
            
            # Time execution
            start_time = time.time()
            final_state = running_streaming_dynamic_eigenvalue_based(
                graph, tesselation, steps, initial_state, angles, tesselation_order,
                matrix_representation='adjacency', searching=[]
            )
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_step = total_time / steps
            
            print(f"  Total time: {total_time:.4f} s")
            print(f"  Time per step: {time_per_step:.4f} s")
            
        except Exception as e:
            print(f"  ERROR: {e}")

def profile_eigenvalue_components():
    """Profile the different components of the eigenvalue implementation"""
    print("=" * 60)
    print("COMPONENT PROFILING")
    print("=" * 60)
    
    N = 100
    steps = 25
    
    try:
        import networkx as nx
        from sqw.tesselations import even_line_two_tesselation
        from sqw.states import uniform_initial_state
        from sqw.utils import random_angle_deviation
        from sqw.experiments_expanded_dynamic_eigenvalue_based import (
            precompute_eigendecompositions, unitary_builder_dynamic
        )
        
        # Setup
        graph = nx.path_graph(N)
        tesselation = even_line_two_tesselation(N)
        initial_state = uniform_initial_state(N, nodes=[N//2])
        angles = random_angle_deviation([math.pi/3, math.pi/3], [0.2, 0.2], steps)
        
        print(f"Profiling components for N={N}, steps={steps}")
        print()
        
        # Profile eigenvalue decomposition
        print("1. Eigenvalue decomposition phase:")
        eig_start = time.time()
        hamiltonians = precompute_eigendecompositions(graph, tesselation, 'adjacency')
        eig_end = time.time()
        eig_time = eig_end - eig_start
        print(f"   Time: {eig_time:.4f} seconds")
        print(f"   Hamiltonians computed: {len(hamiltonians)}")
        
        # Profile unitary building (for one step)
        print("\n2. Unitary building (per step):")
        unitary_start = time.time()
        unitary_operators = unitary_builder_dynamic(hamiltonians, angles[0])
        unitary_end = time.time()
        unitary_time = unitary_end - unitary_start
        print(f"   Time per step: {unitary_time:.6f} seconds")
        print(f"   Unitary operators: {len(unitary_operators)}")
        
        # Estimate total time
        estimated_evolution_time = steps * unitary_time * 2  # Factor of 2 for state evolution
        total_estimated = eig_time + estimated_evolution_time
        
        print(f"\n3. Estimated performance:")
        print(f"   Eigendecomposition: {eig_time:.4f} s (one-time cost)")
        print(f"   Evolution per step: {unitary_time*2:.6f} s")
        print(f"   Total estimated: {total_estimated:.4f} s")
        
        # Show where time is spent
        print(f"\n4. Time breakdown:")
        print(f"   Eigendecomposition: {100*eig_time/total_estimated:.1f}%")
        print(f"   Evolution: {100*estimated_evolution_time/total_estimated:.1f}%")
        
    except Exception as e:
        print(f"ERROR in component profiling: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("Dynamic Quantum Walk Eigenvalue Implementation Analysis")
    print("=" * 60)
    print()
    
    # Test basic performance
    basic_time = test_eigenvalue_performance()
    
    # Test scaling
    test_scaling_performance()
    
    # Profile components  
    profile_eigenvalue_components()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    if basic_time is not None:
        print(f"Basic test (N=100, steps=25): {basic_time:.4f} seconds")
        print(f"Performance for production (N=20000, steps=5000):")
        
        # Estimate production performance
        # Eigenvalue decomposition scales as O(N^3), evolution as O(N^2 * steps)
        scaling_factor = (20000/100)**2  # Conservative N^2 scaling
        step_scaling = 5000/25
        estimated_production = basic_time * scaling_factor * step_scaling
        
        print(f"  Estimated time: {estimated_production:.1f} seconds ({estimated_production/60:.1f} minutes)")
        
        if estimated_production > 3600:  # More than 1 hour
            print("  ⚠️  This suggests the implementation may still be too slow for production!")
        else:
            print("  ✅ Should be acceptable for production use.")

if __name__ == "__main__":
    main()
