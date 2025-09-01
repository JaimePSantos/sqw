#!/usr/bin/env python3

"""
Production Parameter Test

This test uses the exact same parameters as the production run
to identify where the bottleneck is.
"""

import time
import math
import numpy as np
import sys
import os

# Add the current directory to the path to import sqw modules
sys.path.insert(0, os.path.dirname(__file__))

def test_production_parameters():
    """Test with actual production parameters"""
    print("=" * 60)
    print("PRODUCTION PARAMETER TEST")
    print("=" * 60)
    
    # Exact production parameters from generate_dynamic_samples_optimized.py
    N = 20000
    steps = N//4  # 5000
    base_theta = math.pi/3
    dev = 0.0  # Start with dev=0 (no noise) - the first case in the log
    
    print(f"Production Parameters:")
    print(f"  N = {N}")
    print(f"  steps = {steps}")
    print(f"  base_theta = {base_theta}")
    print(f"  deviation = {dev}")
    print()
    
    try:
        # Setup
        import networkx as nx
        from sqw.tesselations import even_line_two_tesselation
        from sqw.states import uniform_initial_state
        from sqw.utils import random_angle_deviation
        from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
        
        print("Setting up environment...")
        setup_start = time.time()
        
        # Create environment (exactly as in the production code)
        graph = nx.path_graph(N)
        tesselation = even_line_two_tesselation(N)
        initial_state = uniform_initial_state(N, nodes=[N//2])
        
        # For dev=0, this should be deterministic
        if dev == 0:
            print("[DEV=0 CASE] No angle noise: Perfect deterministic evolution")
            angles = [[base_theta, base_theta] for _ in range(steps)]
        else:
            angles = random_angle_deviation([base_theta, base_theta], [dev, dev], steps)
        
        tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
        
        setup_end = time.time()
        print(f"Setup completed in {setup_end - setup_start:.4f} seconds")
        print(f"Graph: {N} nodes")
        print(f"Tesselations: {len(tesselation)}")
        print(f"Steps: {steps}")
        print()
        
        # Start timing the actual computation
        print("Starting quantum walk computation...")
        print("WARNING: This may take a very long time with production parameters!")
        print("You may want to interrupt (Ctrl+C) if it takes more than a few minutes.")
        print()
        
        computation_start = time.time()
        
        # Run with a progress callback to see what's happening
        def progress_callback(step, state):
            if step % 100 == 0:
                elapsed = time.time() - computation_start
                print(f"  Step {step}/{steps} completed ({elapsed:.1f}s elapsed)")
        
        final_state = running_streaming_dynamic_eigenvalue_based(
            graph, tesselation, steps, initial_state, angles, tesselation_order,
            matrix_representation='adjacency', searching=[], step_callback=progress_callback
        )
        
        computation_end = time.time()
        total_time = computation_end - computation_start
        
        print(f"\nCOMPUTATION COMPLETED!")
        print(f"Total computation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Time per step: {total_time/steps:.4f} seconds")
        print(f"Final state norm: {np.linalg.norm(final_state):.6f}")
        
        # Estimate time for full sample generation
        print(f"\nESTIMATED FULL SAMPLE GENERATION TIME:")
        print(f"  Time per sample: {total_time:.1f} seconds")
        print(f"  Time for 40 samples: {total_time * 40:.1f} seconds ({total_time * 40 / 3600:.1f} hours)")
        print(f"  Time for all 5 deviations: {total_time * 40 * 5:.1f} seconds ({total_time * 40 * 5 / 3600:.1f} hours)")
        
        return total_time
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Test was interrupted by user")
        print("This suggests the computation would take too long for production use.")
        return None
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_smaller_production_scale():
    """Test with scaled-down but still large parameters"""
    print("=" * 60)
    print("SCALED PRODUCTION TEST")
    print("=" * 60)
    
    # Scaled parameters - still large but manageable
    test_cases = [
        (1000, 250),   # 1K nodes, 250 steps
        (2000, 500),   # 2K nodes, 500 steps  
        (5000, 1250),  # 5K nodes, 1250 steps
    ]
    
    for N, steps in test_cases:
        print(f"\nTesting N={N}, steps={steps}:")
        
        try:
            import networkx as nx
            from sqw.tesselations import even_line_two_tesselation
            from sqw.states import uniform_initial_state
            from sqw.experiments_expanded_dynamic_eigenvalue_based import running_streaming_dynamic_eigenvalue_based
            
            # Setup
            graph = nx.path_graph(N)
            tesselation = even_line_two_tesselation(N)
            initial_state = uniform_initial_state(N, nodes=[N//2])
            angles = [[math.pi/3, math.pi/3] for _ in range(steps)]  # No noise for speed
            tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
            
            # Time the computation
            start_time = time.time()
            final_state = running_streaming_dynamic_eigenvalue_based(
                graph, tesselation, steps, initial_state, angles, tesselation_order,
                matrix_representation='adjacency', searching=[]
            )
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_step = total_time / steps
            
            print(f"  Time: {total_time:.2f}s ({time_per_step:.4f}s/step)")
            
            # Extrapolate to production scale
            production_scaling = (20000/N)**2 * (5000/steps)
            estimated_production_time = total_time * production_scaling
            
            print(f"  Extrapolated production time: {estimated_production_time:.0f}s ({estimated_production_time/3600:.1f}h)")
            
        except Exception as e:
            print(f"  ERROR: {e}")

def main():
    """Main test function"""
    print("Production Parameter Analysis")
    print("=" * 60)
    print()
    
    # First, test smaller scales to understand the scaling
    test_smaller_production_scale()
    
    # Then optionally test full production (user can interrupt)
    print("\n" + "=" * 60)
    response = input("Do you want to test full production parameters? (y/N): ").strip().lower()
    
    if response == 'y':
        test_production_parameters()
    else:
        print("Skipping full production test (wise choice given the scaling results above)")

if __name__ == "__main__":
    main()
