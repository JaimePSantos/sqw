#!/usr/bin/env python3
"""
Simple main interface for static noise quantum walk experiments.

This is the main entry point that provides a clean, simple interface for running
static noise quantum walk experiments with configurable parameters.

Usage:
    python main_static_experiment.py

Customize the experiment by modifying the parameters in the main() function.
"""

import multiprocessing as mp
from experiment_config import ExperimentConfig
from experiment_orchestrator import run_static_experiment


def main():
    """
    Main function to run static noise quantum walk experiments.
    
    Customize the parameters below to control your experiment:
    """
    
    # Create experiment configuration
    config = ExperimentConfig(
        # Core experiment parameters
        N=20000,                    # System size
        samples=20,                 # Samples per deviation
        theta=3.14159/3,           # Base theta parameter (pi/3)
        
        # Deviation values for static noise experiments
        # Format: List of tuples (min_dev, max_dev) or single values
        devs=[
            (0, 0),                 # No noise
            (0, 0.2),              # Small noise range
            (0, 0.6),              # Medium noise range  
            (0, 0.8),              # Medium-high noise range  
            (0, 1),                # High noise range  
        ],
        
        # Execution mode switches
        enable_plotting=True,               # Enable/disable plotting
        use_loglog_plot=True,              # Use log-log scale for std plots
        plot_final_probdist=True,          # Plot final probability distributions
        save_figures=True,                 # Save plots to files
        
        # Archive settings
        create_tar_archive=False,          # Create tar archive of results
        use_multiprocess_archiving=True,   # Use multiprocessing for archiving
        exclude_samples_from_archive=True, # Exclude raw samples from archive
        
        # Computation control - choose your execution mode:
        calculate_samples_only=False,      # Set True to only compute samples
        skip_sample_computation=True,      # Set True to skip sample computation (analysis only)
        
        # Background execution
        run_in_background=False,           # Run in background process
        
        # Multiprocessing settings (None = auto-detect)
        max_processes=None,                # Max processes for sample computation
        use_multiprocess_mean_prob=True,   # Use multiprocessing for mean probability calculation
        max_mean_prob_processes=None,      # Max processes for mean probability
        max_archive_processes=None,        # Max processes for archiving
    )
    
    # Print experiment information
    print("=" * 60)
    print("STATIC NOISE QUANTUM WALK EXPERIMENT")
    print("=" * 60)
    print("This experiment simulates quantum walks with static noise and analyzes")
    print("the spreading behavior through standard deviation calculations.")
    print("")
    print("Experiment phases:")
    print("1. Sample Computation: Generate quantum walk samples for each noise level")
    print("2. Analysis: Calculate mean probability distributions and standard deviations")
    print("3. Plotting: Create visualization plots")
    print("4. Archiving: Create compressed archives of results")
    print("")
    
    # Run the experiment
    try:
        results = run_static_experiment(config)
        
        # Print final results summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Execution mode: {results.get('mode', 'unknown')}")
        print(f"Total time: {results.get('total_time', 0):.2f} seconds")
        
        if 'completed_samples' in results:
            print(f"Samples computed: {results['completed_samples']}")
        
        if 'master_log_file' in results:
            print(f"Master log: {results['master_log_file']}")
        
        if 'archive_name' in results and results['archive_name']:
            print(f"Archive created: {results['archive_name']}")
        
        print("\nExperiment data saved in:")
        print("- experiments_data_samples/          (raw sample data)")
        print("- experiments_data_samples_probDist/ (mean probability distributions)")
        print("- experiments_data_samples_std/      (standard deviation data)")
        
        return results
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user (Ctrl+C)")
        print("Partial results may be available in experiment data directories.")
        return None
        
    except Exception as e:
        print(f"\n\nEXPERIMENT FAILED: {e}")
        print("Check the log files for detailed error information.")
        raise


def quick_test_experiment():
    """
    Quick test experiment with smaller parameters for testing.
    
    Use this function to quickly test the setup with smaller parameters.
    """
    print("Running QUICK TEST experiment with small parameters...")
    
    config = ExperimentConfig(
        N=1000,                    # Small system size for quick testing
        samples=5,                 # Few samples for quick testing
        theta=3.14159/3,          # pi/3
        devs=[(0, 0), (0, 0.2)],  # Just two deviation values
        
        # Enable all features for testing
        enable_plotting=True,
        use_loglog_plot=True,
        plot_final_probdist=True,
        save_figures=True,
        
        # Skip archiving for quick test
        create_tar_archive=False,
        
        # Full pipeline for testing
        calculate_samples_only=False,
        skip_sample_computation=False,
        
        # Use fewer processes for small test
        max_processes=2,
        use_multiprocess_mean_prob=True,
        max_mean_prob_processes=2,
    )
    
    return run_static_experiment(config)


def analysis_only_experiment():
    """
    Run analysis only on existing sample data.
    
    Use this when you have already computed samples and want to rerun
    the analysis with different parameters.
    """
    print("Running ANALYSIS ONLY experiment...")
    
    config = ExperimentConfig(
        # Use the same parameters as your original computation
        N=20000,
        samples=20,
        theta=3.14159/3,
        devs=[(0, 0), (0, 0.2), (0, 0.6), (0, 0.8), (0, 1)],
        
        # Enable analysis and plotting
        enable_plotting=True,
        use_loglog_plot=True,
        plot_final_probdist=True,
        save_figures=True,
        
        # Skip sample computation, enable archiving
        calculate_samples_only=False,
        skip_sample_computation=True,
        create_tar_archive=True,
        
        # Use multiprocessing for analysis
        use_multiprocess_mean_prob=True,
    )
    
    return run_static_experiment(config)


def samples_only_experiment():
    """
    Run sample computation only, skip analysis.
    
    Use this for expensive sample computation that you want to run
    on a cluster, then analyze locally later.
    """
    print("Running SAMPLES ONLY experiment...")
    
    config = ExperimentConfig(
        N=20000,
        samples=20,
        theta=3.14159/3,
        devs=[(0, 0), (0, 0.2), (0, 0.6), (0, 0.8), (0, 1)],
        
        # Only compute samples
        calculate_samples_only=True,
        skip_sample_computation=False,
        
        # Disable plotting and archiving
        enable_plotting=False,
        create_tar_archive=False,
    )
    
    return run_static_experiment(config)


if __name__ == "__main__":
    # Multiprocessing protection for Windows
    mp.set_start_method('spawn', force=True)
    
    # Choose which experiment to run:
    
    # 1. Full experiment (default)
    main()
    
    # 2. Quick test with small parameters
    # quick_test_experiment()
    
    # 3. Analysis only (requires existing sample data)
    # analysis_only_experiment()
    
    # 4. Samples only (for cluster computation)
    # samples_only_experiment()
