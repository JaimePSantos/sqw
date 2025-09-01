#!/usr/bin/env python3

"""
Quick test to verify that the logger fix works
"""

import sys
import os
import logging

# Add the sqw directory to the Python path
sys.path.insert(0, r"c:\Users\jaime\Documents\GitHub\sqw")

# Setup test logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_logger")

try:
    from sqw.experiments_expanded_dynamic_sparse import running_streaming_dynamic_optimized_structure
    print("✓ Successfully imported the optimized function")
    
    # Test that function accepts logger parameter without error
    import networkx as nx
    import numpy as np
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    
    # Small test case
    N = 10
    steps = 5
    graph = nx.path_graph(N)
    tesselation = even_line_two_tesselation(N)
    initial_state = uniform_initial_state(N, nodes=[N//2])
    angles = [[np.pi/6, np.pi/6] for _ in range(steps)]
    tesselation_order = [list(range(len(tesselation))) for _ in range(steps)]
    
    print("Running test with logger...")
    final_state = running_streaming_dynamic_optimized_structure(
        graph, tesselation, steps, initial_state, angles, tesselation_order,
        matrix_representation='adjacency', searching=[], logger=logger
    )
    
    print("✓ Test completed successfully!")
    print(f"Final state shape: {final_state.shape}")
    print(f"State norm: {np.linalg.norm(final_state):.6f}")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
