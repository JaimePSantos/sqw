#!/usr/bin/env python3
"""
Simple test of the new deviation format without multiprocessing
"""
import sys
import os
import math
import numpy as np

# Test the smart loading functions directly
from smart_loading_static import (
    format_deviation_for_filename, 
    get_experiment_dir,
    create_mean_probability_distributions,
    load_mean_probability_distributions,
    check_mean_probability_distributions_exist
)
from sqw.experiments_expanded_static import running

def test_simple_integration():
    """Test the new deviation format with core functions"""
    
    print("Testing Simple Integration (no multiprocessing)")
    print("=" * 50)
    
    # Test parameters - small and simple
    N = 20
    theta = math.pi / 4
    steps = 5
    initial_nodes = [N // 2]  # Center node
    samples = 2
    
    # Test deviation formats
    test_devs = [
        0,              # No noise
        (0.1, 0.0),     # Range [0.0, 0.1] - new format
        (0.2, 0.5),     # Range [0.1, 0.2] - new format
    ]
    
    print(f"Test parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Initial nodes: {initial_nodes}")
    print(f"Test deviations: {test_devs}")
    print("")
    
    # Test 1: Directory creation and naming
    print("1. Testing directory creation and file naming:")
    def dummy_tesselation_func(N):
        return None
    dummy_tesselation_func.__name__ = "dummy_tesselation"
    
    for dev in test_devs:
        # Handle new deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            if dev[1] <= 1.0 and dev[1] >= 0.0:
                max_dev, min_factor = dev
                has_noise = max_dev > 0
            else:
                min_val, max_val = dev
                has_noise = max_val > 0
        else:
            has_noise = dev > 0
        
        noise_params = [dev] if has_noise else [0]
        exp_dir = get_experiment_dir(
            dummy_tesselation_func, has_noise, N,
            noise_params=noise_params, 
            noise_type="static_noise",
            base_dir="test_simple_experiments"
        )
        print(f"  {dev} -> {exp_dir}")
    
    # Test 2: Quantum walk execution
    print("\n2. Testing quantum walk execution:")
    results = []
    for dev in test_devs:
        print(f"  Testing dev={dev}:")
        try:
            probabilities = running(
                N, theta, steps,
                initial_nodes=initial_nodes,
                deviation_range=dev,
                return_all_states=False
            )
            prob_sum = np.sum(probabilities)
            print(f"    SUCCESS! Final probabilities sum: {prob_sum:.6f}")
            results.append(probabilities)
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append(None)
    
    # Test 3: Create sample data using single-threaded approach
    print("\n3. Testing sample data creation (single-threaded):")
    try:
        from smart_loading_static import run_and_save_experiment_samples
        
        # Create dummy angle lists for compatibility
        angles_list_list = []
        for dev_idx, dev in enumerate(test_devs):
            dev_angles = []
            for sample_idx in range(samples):
                dev_angles.append([theta])  # Single theta value
            angles_list_list.append(dev_angles)
        
        # Create initial state function
        def dummy_initial_state_func(N, **kwargs):
            return None  # Not used in static noise
        
        print(f"  Creating samples for {len(test_devs)} deviations with {samples} samples each...")
        
        run_and_save_experiment_samples(
            graph_func=lambda n: None,
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            angles_list_list=angles_list_list,
            tesselation_order=None,
            initial_state_func=dummy_initial_state_func,
            initial_state_kwargs={"nodes": initial_nodes},
            devs=test_devs,
            samples=samples,
            base_dir="test_simple_experiments_samples"
        )
        
        print("  SUCCESS! Sample data created.")
        
    except Exception as e:
        print(f"  ERROR in sample creation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Create mean probability distributions
    print("\n4. Testing mean probability distribution creation:")
    try:
        create_mean_probability_distributions(
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            devs=test_devs,
            samples=samples,
            source_base_dir="test_simple_experiments_samples",
            target_base_dir="test_simple_experiments_probDist",
            noise_type="static_noise"
        )
        print("  SUCCESS! Mean probability distributions created.")
        
    except Exception as e:
        print(f"  ERROR in mean prob dist creation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Load mean probability distributions
    print("\n5. Testing mean probability distribution loading:")
    try:
        # Check if files exist
        files_exist = check_mean_probability_distributions_exist(
            tesselation_func=dummy_tesselation_func,
            N=N,
            steps=steps,
            devs=test_devs,
            base_dir="test_simple_experiments_probDist",
            noise_type="static_noise"
        )
        
        if files_exist:
            mean_results = load_mean_probability_distributions(
                tesselation_func=dummy_tesselation_func,
                N=N,
                steps=steps,
                devs=test_devs,
                base_dir="test_simple_experiments_probDist",
                noise_type="static_noise"
            )
            print(f"  SUCCESS! Loaded {len(mean_results)} deviation results.")
            
            # Print final std values as verification
            for i, (dev, mean_probs) in enumerate(zip(test_devs, mean_results)):
                if mean_probs and len(mean_probs) > 0:
                    final_prob = mean_probs[-1]  # Final time step
                    if final_prob is not None:
                        domain = np.arange(N) - N//2
                        std_val = np.sqrt(np.sum(domain**2 * final_prob.flatten()))
                        print(f"    Dev {dev}: final std = {std_val:.4f}")
        else:
            print("  Files don't exist - this is expected if sample creation failed.")
            
    except Exception as e:
        print(f"  ERROR in loading: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("SIMPLE INTEGRATION TEST COMPLETED")
    print("="*50)

if __name__ == "__main__":
    test_simple_integration()
