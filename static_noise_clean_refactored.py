#!/usr/bin/env python3
"""
Clean static noise experiment runner - Refactored version
This version can be run manually with parameters and uses proper deviation ranges
"""

import argparse
import time
import numpy as np
import os
import sys

# ========================================================================================
# CONFIGURATION SECTION - Edit these parameters to run without command line arguments
# ========================================================================================

# Default experiment parameters (used when no command line args provided)
DEFAULT_CONFIG = {
    'N': 20,                    # Number of nodes
    'theta': np.pi/4,           # Base theta parameter (π/4 ≈ 0.7854)
    'steps': 5,                 # Number of evolution steps (None = auto: N//4)
    'samples': 3,               # Number of samples per deviation range
    'init_nodes': None,         # Initial nodes (None = auto: [N//2])
    'deviation_ranges': [       # Deviation ranges as (min, max) tuples
        (0.0, 0.0),            # No noise
        (0.0, 0.1),            # Positive noise 0 to 0.1
        (0.0, 0.2),            # Positive noise 0 to 0.2
    ],
    'run_analysis': True,       # Whether to run full analysis and plotting
    'save_results': False       # Whether to save results to files
}

# Quick preset configurations - uncomment one to use
# DEFAULT_CONFIG = {
#     'N': 100, 'theta': np.pi/3, 'steps': 20, 'samples': 50,
#     'deviation_ranges': [(0.0, 0.0), (0.0, 0.05), (0.0, 0.1), (0.0, 0.15), (0.0, 0.2)],
#     'run_analysis': True, 'save_results': True
# }

# DEFAULT_CONFIG = {
#     'N': 20, 'theta': np.pi/4, 'steps': 5, 'samples': 3,
#     'deviation_ranges': [(0.0, 0.0), (-0.1, 0.1), (0.0, 0.2)],
#     'run_analysis': False, 'save_results': False
# }

# ========================================================================================

def quick_config(**kwargs):
    """
    Helper function to quickly modify the default configuration
    
    Usage:
    DEFAULT_CONFIG = quick_config(N=100, samples=50, deviation_ranges=[(0.0, 0.0), (0.0, 0.1)])
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config

# Example usage:
# DEFAULT_CONFIG = quick_config(N=100, samples=20, run_analysis=False)

# ========================================================================================

# Import from our existing files
# TODO: Check imports.
from static_noise_production import (
    run_static_noise_samples_experiment,
    create_static_noise_probdist_from_samples,
    save_static_noise_results
)
from jaime_scripts import prob_distributions2std
import matplotlib.pyplot as plt


def create_std_vs_time_plot(mean_results, deviation_values, N, theta, steps, samples):
    """
    Create a simple standard deviation vs time plot for each deviation range
    
    Parameters:
    - mean_results: list of mean probability results for each deviation
    - deviation_values: list of deviation values
    - N: number of nodes
    - theta: base theta parameter
    - steps: number of time steps
    - samples: number of samples
    """
    print("📊 Calculating standard deviations for each deviation range...")
    
    # Calculate standard deviations
    std_data = {}
    time_points = list(range(steps))
    
    for i, dev_val in enumerate(deviation_values):
        if i < len(mean_results) and mean_results[i] is not None:
            # mean_results[i] should be a list of probability distributions for each time step
            if isinstance(mean_results[i], list):
                # Multiple time steps
                std_values = []
                for step_probs in mean_results[i]:
                    std_val = prob_distributions2std(step_probs, N)
                    std_values.append(std_val)
                std_data[dev_val] = std_values
            else:
                # Single time step (final result only)
                std_val = prob_distributions2std(mean_results[i], N)
                std_data[dev_val] = [std_val]  # Make it a list for consistency
        else:
            print(f"⚠️  No data for deviation {dev_val}")
            std_data[dev_val] = [0.0] * steps
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot standard deviation vs time for each deviation range
    colors = plt.cm.viridis(np.linspace(0, 1, len(deviation_values)))
    
    for i, dev_val in enumerate(deviation_values):
        if dev_val in std_data and len(std_data[dev_val]) > 0:
            # Adjust time points to match data length
            actual_steps = len(std_data[dev_val])
            x_points = list(range(actual_steps))
            
            plt.plot(x_points, std_data[dev_val], 
                    'o-', color=colors[i], linewidth=2, markersize=6,
                    label=f'Deviation {dev_val:.3f}')
    
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.title(f'Standard Deviation vs Time\nN={N}, θ={theta:.4f}, {samples} samples', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    if steps <= 20:
        plt.xticks(range(steps))
    
    plt.tight_layout()
    print("📈 Displaying standard deviation vs time plot...")
    plt.show()
    
    # Print summary
    print("\n📋 Standard Deviation Summary:")
    for dev_val in deviation_values:
        if dev_val in std_data and len(std_data[dev_val]) > 0:
            final_std = std_data[dev_val][-1]
            if isinstance(final_std, (list, np.ndarray)):
                final_std = float(final_std[0]) if len(final_std) > 0 else 0.0
            print(f"   Deviation {dev_val:.3f}: Final std = {final_std:.4f}")
        else:
            print(f"   Deviation {dev_val:.3f}: No data available")


def run_static_noise_experiment_clean(N=50, theta=np.pi/4, steps=None, samples=10, 
                                     init_nodes=None, deviation_ranges=None, 
                                     run_analysis=True, save_results=True):
    """
    Run static noise staggered quantum walk experiment with clean interface
    
    Parameters:
    - N: number of nodes (default: 50)
    - theta: base theta parameter (default: π/4)
    - steps: number of evolution steps (default: N//4)
    - samples: number of samples per deviation range (default: 10)
    - init_nodes: list of initial nodes (default: [N//2])
    - deviation_ranges: list of (min, max) tuples for noise ranges (default: [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2)])
    - run_analysis: whether to run full analysis and plotting (default: True)
    - save_results: whether to save results to files (default: True)
    
    Returns:
    - Dictionary with experiment results
    """
    
    print("🔬 Static Noise Staggered Quantum Walk - Clean Version")
    print("=" * 60)
    
    # Set defaults
    if steps is None:
        steps = max(5, N//4)
    if init_nodes is None:
        init_nodes = [N//2]
    if deviation_ranges is None:
        deviation_ranges = [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2)]
    
    print(f"🎯 Experiment parameters:")
    print(f"   N = {N} (quantum walk nodes)")
    print(f"   theta = {theta:.4f} (≈ π/{np.pi/theta:.1f})")
    print(f"   steps = {steps} (time evolution)")
    print(f"   samples = {samples} (per deviation range)")
    print(f"   init_nodes = {init_nodes} (starting position)")
    print(f"   deviation_ranges = {deviation_ranges}")
    print(f"   Total quantum walks: {len(deviation_ranges)} × {samples} = {len(deviation_ranges) * samples}")
    print()
    
    start_time = time.time()
    
    # Convert deviation ranges to single values for compatibility with existing code
    # Take the maximum absolute value from each range as the deviation parameter
    deviation_values = []
    for dev_min, dev_max in deviation_ranges:
        if dev_min == 0.0 and dev_max == 0.0:
            deviation_values.append(0.0)  # No noise case
        else:
            # Use the larger absolute value as the deviation range
            deviation_values.append(max(abs(dev_min), abs(dev_max)))
    
    print(f"📊 Converted deviation ranges to values: {deviation_values}")
    print("   (Each value creates uniform noise in [-value, +value] range)")
    print()
    
    # Run the experiment using our existing production function
    try:
        print("🚀 Running static noise experiment...")
        mean_results = run_static_noise_samples_experiment(
            N, theta, steps, init_nodes, deviation_values, samples
        )
        
        # Create mean probability distributions if needed
        print("📊 Creating mean probability distributions...")
        create_static_noise_probdist_from_samples(N, theta, steps, deviation_values, samples)
        
        experiment_time = time.time() - start_time
        print(f"✅ Experiment completed in {experiment_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Analysis and visualization
    if run_analysis and mean_results:
        print("\n📈 Creating standard deviation vs time plot...")
        try:
            # Calculate standard deviations for each deviation range
            create_std_vs_time_plot(
                mean_results, deviation_values, N, theta, steps, samples
            )
            
            # Save detailed results if requested
            if save_results:
                save_static_noise_results(
                    {'deviation_values': deviation_values, 'mean_results': mean_results}, 
                    deviation_values, N, theta, steps, samples
                )
            
            print("✅ Standard deviation analysis completed!")
            
        except Exception as e:
            print(f"⚠️  Standard deviation analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠️  Skipping analysis (run_analysis=False or no results)")
    
    # Remove the old std analysis section since we're replacing it
    print("\n📊 Standard deviation analysis completed!")
    
    # Performance summary
    print("\n=== Experiment Summary ===")
    print(f"System size (N): {N}")
    print(f"Base theta: {theta:.4f}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Deviation ranges tested: {len(deviation_ranges)}")
    print(f"Total quantum walks: {len(deviation_ranges) * samples}")
    if experiment_time > 0:
        print(f"Average time per quantum walk: {experiment_time / (len(deviation_ranges) * samples):.3f} seconds")
    
    return {
        "N": N,
        "theta": theta,
        "steps": steps,
        "samples": samples,
        "init_nodes": init_nodes,
        "deviation_ranges": deviation_ranges,
        "deviation_values": deviation_values,
        "mean_results": mean_results if 'mean_results' in locals() else None,
        "total_time": experiment_time,
        "success": True
    }


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(
        description='Run static noise staggered quantum walk experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python static_noise_clean_refactored.py --N 50 --steps 10 --samples 5
  python static_noise_clean_refactored.py --N 100 --theta 0.7854 --deviation-ranges "(0.0,0.0)" "(0.0,0.1)" "(0.0,0.2)"
  python static_noise_clean_refactored.py --N 20 --samples 3 --no-analysis
  
To run with default parameters set in the code, simply run:
  python static_noise_clean_refactored.py
        """
    )
    
    # Check if any arguments were provided
    args_provided = len(sys.argv) > 1
    
    parser.add_argument('--N', type=int, default=DEFAULT_CONFIG['N'],
                       help=f'Number of nodes (default: {DEFAULT_CONFIG["N"]})')
    parser.add_argument('--theta', type=float, default=DEFAULT_CONFIG['theta'],
                       help=f'Base theta parameter (default: {DEFAULT_CONFIG["theta"]:.4f})')
    parser.add_argument('--steps', type=int, default=DEFAULT_CONFIG['steps'],
                       help=f'Number of evolution steps (default: {DEFAULT_CONFIG["steps"] if DEFAULT_CONFIG["steps"] else "N//4"})')
    parser.add_argument('--samples', type=int, default=DEFAULT_CONFIG['samples'],
                       help=f'Number of samples per deviation range (default: {DEFAULT_CONFIG["samples"]})')
    parser.add_argument('--init-nodes', nargs='+', type=int, default=DEFAULT_CONFIG['init_nodes'],
                       help=f'Initial nodes (default: {DEFAULT_CONFIG["init_nodes"] if DEFAULT_CONFIG["init_nodes"] else "[N//2]"})')
    parser.add_argument('--deviation-ranges', nargs='*', default=None,
                       help=f'Deviation ranges as "(min,max)" tuples (default: {DEFAULT_CONFIG["deviation_ranges"]})')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip analysis and plotting')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to files')
    
    args = parser.parse_args()
    
    # Use default configuration when no arguments provided
    if not args_provided:
        print("🔧 No command line arguments provided - using DEFAULT_CONFIG from code")
        print("   (Edit DEFAULT_CONFIG in the script to change parameters)")
        print()
        
        # Override with default config values
        deviation_ranges = DEFAULT_CONFIG['deviation_ranges']
        init_nodes = DEFAULT_CONFIG['init_nodes']
        run_analysis = DEFAULT_CONFIG['run_analysis']
        save_results = DEFAULT_CONFIG['save_results']
        
        # Use config values directly
        final_args = {
            'N': DEFAULT_CONFIG['N'],
            'theta': DEFAULT_CONFIG['theta'], 
            'steps': DEFAULT_CONFIG['steps'],
            'samples': DEFAULT_CONFIG['samples']
        }
        
    else:
        # Parse deviation ranges from command line
        if args.deviation_ranges is None:
            deviation_ranges = DEFAULT_CONFIG['deviation_ranges']
        else:
            deviation_ranges = []
            for range_str in args.deviation_ranges:
                try:
                    # Parse "(min,max)" format
                    range_str = range_str.strip('()')
                    min_val, max_val = map(float, range_str.split(','))
                    deviation_ranges.append((min_val, max_val))
                except Exception as e:
                    print(f"Error parsing deviation range '{range_str}': {e}")
                    print("Use format: '(min,max)' for example '(0.0,0.1)'")
                    sys.exit(1)
        
        # Set init_nodes default if not provided
        if args.init_nodes is None:
            init_nodes = [args.N // 2]
        else:
            init_nodes = args.init_nodes
            
        run_analysis = not args.no_analysis
        save_results = not args.no_save
        
        final_args = {
            'N': args.N,
            'theta': args.theta,
            'steps': args.steps,
            'samples': args.samples
        }
    
    # Set init_nodes default if still None
    if init_nodes is None:
        init_nodes = [final_args['N'] // 2]
    
    print("🎯 Experiment configuration:")
    print(f"   N = {final_args['N']}")
    print(f"   theta = {final_args['theta']:.4f}")
    print(f"   steps = {final_args['steps'] if final_args['steps'] else 'auto'}")
    print(f"   samples = {final_args['samples']}")
    print(f"   init_nodes = {init_nodes}")
    print(f"   deviation_ranges = {deviation_ranges}")
    print(f"   run_analysis = {run_analysis}")
    print(f"   save_results = {save_results}")
    print()
    
    # Run the experiment
    result = run_static_noise_experiment_clean(
        N=final_args['N'],
        theta=final_args['theta'],
        steps=final_args['steps'],
        samples=final_args['samples'],
        init_nodes=init_nodes,
        deviation_ranges=deviation_ranges,
        run_analysis=run_analysis,
        save_results=save_results
    )
    
    if result and result.get('success', False):
        print("\n🎉 Experiment completed successfully!")
        return 0
    else:
        print("\n❌ Experiment failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
