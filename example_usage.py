#!/usr/bin/env python3
"""
Example usage of the modular static experiment package.

This demonstrates how simple it is to use the new modular interface.
"""

import multiprocessing as mp
from static_experiment_modules import ExperimentConfig, run_static_experiment


def quick_example():
    """Quick example with minimal configuration."""
    print("Running a quick test experiment...")
    
    # Create a simple configuration
    config = ExperimentConfig(
        N=1000,                    # Small system for quick test
        samples=3,                 # Just a few samples
        devs=[(0, 0), (0, 0.1)],  # Two noise levels
        enable_plotting=True,      # Show plots
        create_tar_archive=False,  # No archiving for test
        max_processes=2            # Limit processes
    )
    
    # Run the experiment
    results = run_static_experiment(config)
    
    print(f"\nExperiment completed in {results['total_time']:.1f} seconds")
    print(f"Mode: {results['mode']}")
    
    return results


def analysis_only_example():
    """Example of running analysis only on existing data."""
    print("Running analysis only on existing data...")
    
    config = ExperimentConfig(
        N=20000,                      # Same as original computation
        samples=20,                   # Same as original computation
        devs=[(0, 0), (0, 0.2), (0, 0.6), (0, 0.8), (0, 1)],
        skip_sample_computation=True, # Skip sample computation
        enable_plotting=True,         # Generate plots
        create_tar_archive=True       # Create archive
    )
    
    results = run_static_experiment(config)
    return results


def custom_experiment():
    """Example with custom parameters."""
    print("Running custom experiment...")
    
    config = ExperimentConfig(
        # Custom experiment parameters
        N=5000,
        samples=10,
        theta=3.14159/4,  # Different theta value
        devs=[(0, 0), (0, 0.3), (0, 0.7)],  # Custom noise levels
        
        # Custom execution settings
        enable_plotting=True,
        use_loglog_plot=False,      # Use linear scale
        plot_final_probdist=False,  # Skip final prob dist plots
        
        # Custom multiprocessing
        max_processes=3,
        use_multiprocess_mean_prob=False,  # Use single process for mean prob
        
        # Background execution
        run_in_background=False
    )
    
    results = run_static_experiment(config)
    return results


if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    print("=" * 60)
    print("STATIC EXPERIMENT MODULE EXAMPLES")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # 1. Quick test
    quick_example()
    
    # 2. Analysis only (requires existing data)
    # analysis_only_example()
    
    # 3. Custom experiment
    # custom_experiment()
    
    print("\nExamples completed successfully!")
    print("See how simple the interface is now - just create a config and run!")
