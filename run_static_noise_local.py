#!/usr/bin/env python3
"""
Static Noise Experiment Runner - Local Version
Run the static noise experiment without cluster deployment for local testing
"""

import time
import numpy as np


def run_local_static_noise_experiment():
    """Run the static noise experiment locally without cluster deployment"""
    
    print("Static Noise Staggered Quantum Walk - Local Version")
    print("=" * 55)
    
    # Import all the functions from the static_noise_clean module
    from static_noise_clean import (
        smart_load_static_noise_experiment,
        run_direct_static_noise_experiment,
        analyze_static_noise_results,
        create_static_noise_plots,
        save_static_noise_results
    )
    
    # Experiment parameters (optimized for local testing)
    N = 20  # Moderate system size for testing
    theta = np.pi / 4  # Base theta parameter
    steps = 10  # Time steps  
    samples = 20  # Samples per deviation
    init_nodes = [N//2]  # Start at center node
    
    # List of static noise deviations to test
    deviation_ranges = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    print(f"Local experiment parameters:")
    print(f"  N = {N}")
    print(f"  theta = π/{4 if theta == np.pi/4 else f'{np.pi/theta:.1f}'}")
    print(f"  steps = {steps}")
    print(f"  samples = {samples}")
    print(f"  init_nodes = {init_nodes}")
    print(f"  deviation_ranges = {deviation_ranges}")
    print()
    
    # Run the experiment
    print("🧠 Using smart load system for efficient data management...")
    start_time = time.time()
    
    try:
        mean_results = smart_load_static_noise_experiment(
            N=N,
            theta=theta,
            steps=steps,
            init_nodes=init_nodes,
            deviation_ranges=deviation_ranges,
            samples=samples
        )
        print("✅ Smart loading completed successfully!")
        
    except Exception as e:
        print(f"⚠️  Smart loading failed: {e}")
        print("📝 Falling back to direct experiment execution...")
        mean_results = run_direct_static_noise_experiment(
            N, theta, steps, init_nodes, deviation_ranges, samples
        )
    
    experiment_time = time.time() - start_time
    print(f"\n🎉 Static noise experiment completed in {experiment_time:.2f} seconds")
    
    # Analysis and visualization
    print("\n📊 Analyzing static noise effects...")
    try:
        analysis_results = analyze_static_noise_results(
            mean_results, deviation_ranges, N, steps
        )
        
        # Create comprehensive plots
        create_static_noise_plots(
            analysis_results, deviation_ranges, N, theta, steps, samples
        )
        
        # Save detailed results
        save_static_noise_results(
            analysis_results, deviation_ranges, N, theta, steps, samples
        )
        
        print("✅ Analysis and visualization completed!")
        
    except Exception as e:
        print(f"⚠️  Analysis failed: {e}")
        print("Results are still available in the experiment directories.")
        import traceback
        traceback.print_exc()
    
    # Performance summary
    print("\n=== Performance Summary ===")
    print(f"System size (N): {N}")
    print(f"Base theta: {theta:.4f}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Number of deviations: {len(deviation_ranges)}")
    print(f"Total quantum walks: {len(deviation_ranges) * samples}")
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(deviation_ranges) * samples):.3f} seconds")
    
    print("\n=== Static Noise Model Details ===")
    print("Static noise model:")
    print("- deviation=0.0: Perfect tessellation (no noise)")
    print("- deviation>0: Random deviations applied to tessellation edges")
    print("- Each sample generates different random noise parameters")
    print("- Red and blue tessellation edges get independent noise")
    print("- Mean probability distributions average over all samples")
    
    return {
        "deviation_ranges": deviation_ranges,
        "N": N,
        "theta": theta,
        "steps": steps,
        "samples": samples,
        "total_time": experiment_time,
        "analysis_results": analysis_results if 'analysis_results' in locals() else None
    }


def run_quick_demo():
    """Run a very quick demo with minimal parameters"""
    
    print("Quick Static Noise Demo")
    print("=" * 25)
    
    from static_noise_clean import run_direct_static_noise_experiment
    
    # Minimal parameters for quick demo
    N = 6
    theta = np.pi / 4
    steps = 3
    init_nodes = [N//2]
    deviation_ranges = [0.0, 0.1, 0.2]
    samples = 5
    
    print(f"Demo parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Deviations: {deviation_ranges}")
    
    start_time = time.time()
    
    mean_results = run_direct_static_noise_experiment(
        N, theta, steps, init_nodes, deviation_ranges, samples
    )
    
    demo_time = time.time() - start_time
    print(f"Demo completed in {demo_time:.2f} seconds")
    
    # Simple analysis
    print("\nResults:")
    for i, (dev, result) in enumerate(zip(deviation_ranges, mean_results)):
        max_prob = np.max(result)
        max_node = np.argmax(result)
        print(f"  Deviation {dev:.1f}: max_prob={max_prob:.4f} at node {max_node}")
    
    return mean_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        # Quick demo mode
        run_quick_demo()
    else:
        # Full experiment
        try:
            result = run_local_static_noise_experiment()
            print("\n✅ Experiment completed successfully!")
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
