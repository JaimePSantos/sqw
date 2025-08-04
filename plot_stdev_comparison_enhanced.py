"""
Enhanced Plot Standard Deviation Comparison for Angle and Tesselation Order Noise

This script loads probability distributions from existing experiments and plots
the standard deviation as a function of time steps for both angle noise and
tesselation order noise experiments. It also saves plots and provides additional
analysis capabilities.
"""

from sqw.tesselations import even_line_two_tesselation
from sqw.states import uniform_initial_state
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from jaime_scripts import (
    load_mean_probability_distributions,
    check_mean_probability_distributions_exist,
    prob_distributions2std,
    plot_std_vs_time_qwak
)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Experiment parameters
N = 3000  # System size
DEFAULT_STEPS = None  # If None, will be calculated as N // 4
SAMPLES = 1  # Number of samples used in experiments
BASE_DIR = "experiments_data_samples_probDist"  # Directory containing experiment data
OUTPUT_DIR = "plot_outputs"  # Directory for saving plots and data

# Angle noise experiment parameters
ANGLE_DEVIATIONS = [0, (np.pi/3)/2.5, (np.pi/3) * 2]

# Tesselation order noise experiment parameters  
SHIFT_PROBABILITIES = [0, 0.2, 0.5]

# Plot styling parameters
PLOT_COLORS = ['blue', 'orange', 'green', 'red', 'purple']
FIGURE_SIZE_COMBINED = (15, 6)
FIGURE_SIZE_INDIVIDUAL = (10, 6)
DPI = 300
MARKER_SIZE = 2
LINE_WIDTH = 1.5

# Grid and axis parameters
GRID_ALPHA = 0.3
AXIS_MARGIN_FACTOR = 0.1  # For regular plots
LOG_AXIS_MARGIN_FACTOR_TIME = 0.2  # For log-log plots (time axis)
LOG_AXIS_MARGIN_FACTOR_STD = 0.5   # For log-log plots (std axis)

# =============================================================================
# CONFIGURATION GUIDE
# =============================================================================
"""
CONFIGURATION GUIDE:

To modify the script behavior, edit the configuration parameters above:

EXPERIMENT PARAMETERS:
- N: System size (default: 2000)
- DEFAULT_STEPS: Number of time steps (default: N//4)
- SAMPLES: Number of samples used in experiments (default: 1)
- BASE_DIR: Directory containing experiment data
- OUTPUT_DIR: Directory for saving plots and data

NOISE PARAMETERS:
- ANGLE_DEVIATIONS: List of angle deviation values to analyze
- SHIFT_PROBABILITIES: List of shift probabilities for tesselation order noise

PLOT STYLING:
- PLOT_COLORS: Colors for different data series
- FIGURE_SIZE_COMBINED: Size for combined plots (width, height)
- FIGURE_SIZE_INDIVIDUAL: Size for individual plots (width, height) 
- DPI: Resolution for saved plots
- MARKER_SIZE: Size of markers in plots
- LINE_WIDTH: Width of lines in plots

AXIS AND GRID:
- GRID_ALPHA: Transparency of grid lines (0-1)
- AXIS_MARGIN_FACTOR: Margin around data for regular plots
- LOG_AXIS_MARGIN_FACTOR_TIME: Margin for time axis in log-log plots
- LOG_AXIS_MARGIN_FACTOR_STD: Margin for std axis in log-log plots

USAGE EXAMPLES:
1. Change system size: N = 3000
2. Add more angle deviations: ANGLE_DEVIATIONS = [0, 0.1, 0.2, 0.5, 1.0]
3. Change number of samples: SAMPLES = 10
4. Higher resolution plots: DPI = 600
5. Larger figures: FIGURE_SIZE_COMBINED = (20, 8)
"""

def save_plot_data(angle_stds, angle_devs, tesselation_stds, shift_probs, N, samples=SAMPLES, output_dir=OUTPUT_DIR):
    """
    Save the plot data to files for later analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save angle data
    if angle_stds and angle_devs:
        angle_data = {
            'deviations': angle_devs,
            'std_values': angle_stds,
            'timesteps': [list(range(len(std))) for std in angle_stds],
            'experiment_type': 'angle_noise',
            'N': N,
            'samples': samples,
            'steps': len(angle_stds[0]) if angle_stds and len(angle_stds[0]) > 0 else 0
        }
        import json
        filename = f'angle_noise_std_data_N{N}_samples{samples}.json'
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(angle_data, f, indent=2)
        print(f"✅ Saved angle noise data to {output_dir}/{filename}")
    
    # Save tesselation data
    if tesselation_stds and shift_probs:
        tesselation_data = {
            'shift_probabilities': shift_probs,
            'std_values': tesselation_stds,
            'timesteps': [list(range(len(std))) for std in tesselation_stds],
            'experiment_type': 'tesselation_order_noise',
            'N': N,
            'samples': samples,
            'steps': len(tesselation_stds[0]) if tesselation_stds and len(tesselation_stds[0]) > 0 else 0
        }
        filename = f'tesselation_order_std_data_N{N}_samples{samples}.json'
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(tesselation_data, f, indent=2)
        print(f"✅ Saved tesselation order data to {output_dir}/{filename}")

def load_and_plot_angle_experiments(N=N, steps=DEFAULT_STEPS, devs=ANGLE_DEVIATIONS, base_dir=BASE_DIR):
    """
    Load angle noise experiments and calculate standard deviations.
    """
    if steps is None:
        steps = N // 4
    
    print(f"Loading angle noise experiments for N={N}, steps={steps}")
    print(f"Angle deviations: {devs}")
    
    # Check if experiments exist
    if not check_mean_probability_distributions_exist(
        even_line_two_tesselation, N, steps, devs, base_dir, "angle"
    ):
        print("❌ Angle noise experiments not found in the expected directory.")
        print(f"Expected directory: {base_dir}")
        return None, None
    
    # Load the mean probability distributions
    results = load_mean_probability_distributions(
        even_line_two_tesselation, N, steps, devs, base_dir, "angle"
    )
    
    # Calculate standard deviations
    domain = np.arange(N)
    stds = []
    
    for i, dev_mean_prob_dists in enumerate(results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Angle dev {i} (dev={devs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Angle dev {i} (dev={devs[i]:.3f}): No valid probability distributions")
            stds.append([])
    
    return stds, devs

def load_and_plot_tesselation_experiments(N=N, steps=DEFAULT_STEPS, shift_probs=SHIFT_PROBABILITIES, base_dir=BASE_DIR):
    """
    Load tesselation order noise experiments and calculate standard deviations.
    """
    if steps is None:
        steps = N // 4
    
    print(f"Loading tesselation order experiments for N={N}, steps={steps}")
    print(f"Shift probabilities: {shift_probs}")
    
    # Check if experiments exist
    if not check_mean_probability_distributions_exist(
        even_line_two_tesselation, N, steps, shift_probs, base_dir, "tesselation_order"
    ):
        print("❌ Tesselation order experiments not found in the expected directory.")
        print(f"Expected directory: {base_dir}")
        return None, None
    
    # Load the mean probability distributions
    results = load_mean_probability_distributions(
        even_line_two_tesselation, N, steps, shift_probs, base_dir, "tesselation_order"
    )
    
    # Calculate standard deviations
    domain = np.arange(N)
    stds = []
    
    for i, shift_mean_prob_dists in enumerate(results):
        if shift_mean_prob_dists and len(shift_mean_prob_dists) > 0 and all(state is not None for state in shift_mean_prob_dists):
            std_values = prob_distributions2std(shift_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Tesselation shift {i} (prob={shift_probs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Tesselation shift {i} (prob={shift_probs[i]:.3f}): No valid probability distributions")
            stds.append([])
    
    return stds, shift_probs

def plot_and_save_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs, 
                                     colors=PLOT_COLORS, figsize=FIGURE_SIZE_COMBINED, 
                                     marker_size=MARKER_SIZE, line_width=LINE_WIDTH, 
                                     grid_alpha=GRID_ALPHA, axis_margin=AXIS_MARGIN_FACTOR,
                                     N=N, samples=SAMPLES, output_dir=OUTPUT_DIR, dpi=DPI):
    """
    Plot both angle and tesselation experiments in a combined figure and save to file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot angle noise results
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        ax1.set_title('Standard Deviation vs Time - Angle Noise', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Standard Deviation', fontsize=12)
        ax1.grid(True, alpha=grid_alpha)
        
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax1.plot(timesteps, std_values, marker='o', markersize=marker_size, 
                        color=colors[i % len(colors)], linewidth=line_width,
                        label=f'angle_dev={dev:.3f}')
        
        ax1.legend()
        ax1.set_xlim(0, max(len(std) for std in angle_stds if len(std) > 0))
        
        # Set Y-axis limits based on data range for angle noise
        if angle_stds and any(len(std) > 0 for std in angle_stds):
            all_stds = []
            for std_values in angle_stds:
                if len(std_values) > 0:
                    all_stds.extend(std_values)
            if all_stds:
                min_std = np.min(all_stds)
                max_std = np.max(all_stds)
                margin = (max_std - min_std) * axis_margin
                ax1.set_ylim(min_std - margin, max_std + margin)
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Standard Deviation vs Time - Angle Noise (No Data)')
    
    # Plot tesselation order results
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Standard Deviation', fontsize=12)
        ax2.grid(True, alpha=grid_alpha)
        
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax2.plot(timesteps, std_values, marker='s', markersize=marker_size, 
                        color=colors[i % len(colors)], linewidth=line_width,
                        label=f'shift_prob={prob:.3f}')
        
        ax2.legend()
        ax2.set_xlim(0, max(len(std) for std in tesselation_stds if len(std) > 0))
        
        # Set Y-axis limits based on data range for tesselation noise
        if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
            all_stds = []
            for std_values in tesselation_stds:
                if len(std_values) > 0:
                    all_stds.extend(std_values)
            if all_stds:
                min_std = np.min(all_stds)
                max_std = np.max(all_stds)
                margin = (max_std - min_std) * axis_margin
                ax2.set_ylim(min_std - margin, max_std + margin)
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, f'std_comparison_combined_N{N}_samples{samples}.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved combined plot to {output_file}")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, f'std_comparison_combined_N{N}_samples{samples}.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined plot to {output_file_pdf}")
    
    plt.show()

def plot_and_save_combined_comparison_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, 
                                           colors=PLOT_COLORS, figsize=FIGURE_SIZE_COMBINED, 
                                           marker_size=MARKER_SIZE, line_width=LINE_WIDTH, 
                                           grid_alpha=GRID_ALPHA, time_margin=LOG_AXIS_MARGIN_FACTOR_TIME,
                                           std_margin=LOG_AXIS_MARGIN_FACTOR_STD, N=N, samples=SAMPLES, 
                                           output_dir=OUTPUT_DIR, dpi=DPI):
    """
    Plot both angle and tesselation experiments in log-log scale to reveal scaling behavior.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot angle noise results (log-log)
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        ax1.set_title('Standard Deviation vs Time - Angle Noise (Log-Log)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step (log scale)', fontsize=12)
        ax1.set_ylabel('Standard Deviation (log scale)', fontsize=12)
        ax1.grid(True, alpha=grid_alpha)
        
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    ax1.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='o', markersize=marker_size, color=colors[i % len(colors)], 
                              linewidth=line_width, label=f'angle_dev={dev:.3f}')
        
        ax1.legend()
        
        # Set axis limits based on data range for angle noise (log-log)
        if angle_stds and any(len(std) > 0 for std in angle_stds):
            all_stds = []
            all_timesteps = []
            for std_values in angle_stds:
                if len(std_values) > 0:
                    std_array = np.array(std_values)
                    timesteps = np.array(range(1, len(std_values) + 1))
                    valid_mask = std_array > 0
                    if np.any(valid_mask):
                        all_stds.extend(std_array[valid_mask])
                        all_timesteps.extend(timesteps[valid_mask])
            if all_stds and all_timesteps:
                min_std = np.min(all_stds)
                max_std = np.max(all_stds)
                min_time = np.min(all_timesteps)
                max_time = np.max(all_timesteps)
                ax1.set_xlim(min_time * (1 - time_margin), max_time * (1 + time_margin))
                ax1.set_ylim(min_std * (1 - std_margin), max_std * (1 + std_margin))
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Standard Deviation vs Time - Angle Noise (Log-Log, No Data)')
    
    # Plot tesselation order results (log-log)
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step (log scale)', fontsize=12)
        ax2.set_ylabel('Standard Deviation (log scale)', fontsize=12)
        ax2.grid(True, alpha=grid_alpha)
        
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    ax2.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='s', markersize=marker_size, color=colors[i % len(colors)], 
                              linewidth=line_width, label=f'shift_prob={prob:.3f}')
        
        ax2.legend()
        
        # Set axis limits based on data range for tesselation noise (log-log)
        if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
            all_stds = []
            all_timesteps = []
            for std_values in tesselation_stds:
                if len(std_values) > 0:
                    std_array = np.array(std_values)
                    timesteps = np.array(range(1, len(std_values) + 1))
                    valid_mask = std_array > 0
                    if np.any(valid_mask):
                        all_stds.extend(std_array[valid_mask])
                        all_timesteps.extend(timesteps[valid_mask])
            if all_stds and all_timesteps:
                min_std = np.min(all_stds)
                max_std = np.max(all_stds)
                min_time = np.min(all_timesteps)
                max_time = np.max(all_timesteps)
                ax2.set_xlim(min_time * (1 - time_margin), max_time * (1 + time_margin))
                ax2.set_ylim(min_std * (1 - std_margin), max_std * (1 + std_margin))
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log, No Data)')
    
    plt.tight_layout()
    
    # Save the log-log plot
    output_file = os.path.join(output_dir, f'std_comparison_combined_loglog_N{N}_samples{samples}.png')
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved combined log-log plot to {output_file}")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, f'std_comparison_combined_loglog_N{N}_samples{samples}.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined log-log plot to {output_file_pdf}")
    
    plt.show()

def plot_individual_experiments_with_save(angle_stds, angle_devs, tesselation_stds, shift_probs, 
                                         colors=PLOT_COLORS, figsize=FIGURE_SIZE_INDIVIDUAL, 
                                         marker_size=MARKER_SIZE, line_width=LINE_WIDTH, 
                                         grid_alpha=GRID_ALPHA, axis_margin=AXIS_MARGIN_FACTOR,
                                         N=N, samples=SAMPLES, output_dir=OUTPUT_DIR, dpi=DPI):
    """
    Plot angle and tesselation experiments separately and save them.
    """
    print("\n" + "="*60)
    print("PLOTTING INDIVIDUAL EXPERIMENTS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot angle noise experiments
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        print("\nPlotting angle noise experiments...")
        
        plt.figure(figsize=figsize)
        
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                plt.plot(timesteps, std_values, marker='o', markersize=marker_size, 
                        color=colors[i % len(colors)], linewidth=line_width,
                        label=f'angle_dev={dev:.3f}')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Standard Deviation vs Time for Different Angle Noise Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=grid_alpha)
        
        # Set Y-axis limits based on data range for angle noise
        all_stds = []
        for std_values in angle_stds:
            if len(std_values) > 0:
                all_stds.extend(std_values)
        if all_stds:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            margin = (max_std - min_std) * axis_margin
            plt.ylim(min_std - margin, max_std + margin)
        
        plt.tight_layout()
        
        # Save angle plot
        angle_output = os.path.join(output_dir, f'std_angle_noise_N{N}_samples{samples}.png')
        plt.savefig(angle_output, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved angle noise plot to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, f'std_angle_noise_N{N}_samples{samples}.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise plot to {angle_output_pdf}")
        
        plt.close()
        
    else:
        print("\nNo angle noise data to plot.")
    
    # Plot tesselation order experiments
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nPlotting tesselation order experiments...")
        
        plt.figure(figsize=figsize)
        
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                plt.plot(timesteps, std_values, marker='s', markersize=marker_size, 
                        color=colors[i % len(colors)], linewidth=line_width,
                        label=f'shift_prob={prob:.3f}')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Standard Deviation vs Time for Different Tesselation Shift Probabilities', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=grid_alpha)
        
        # Set Y-axis limits based on data range for tesselation noise
        all_stds = []
        for std_values in tesselation_stds:
            if len(std_values) > 0:
                all_stds.extend(std_values)
        if all_stds:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            margin = (max_std - min_std) * axis_margin
            plt.ylim(min_std - margin, max_std + margin)
        
        plt.tight_layout()
        
        # Save tesselation plot
        tess_output = os.path.join(output_dir, f'std_tesselation_order_N{N}_samples{samples}.png')
        plt.savefig(tess_output, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved tesselation order plot to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, f'std_tesselation_order_N{N}_samples{samples}.pdf')
        plt.savefig(tess_output_pdf, bbox_inches='tight')
        print(f"✅ Saved tesselation order plot to {tess_output_pdf}")
        
        plt.close()
    else:
        print("\nNo tesselation order data to plot.")

def plot_individual_experiments_with_save_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, 
                                               colors=PLOT_COLORS, figsize=FIGURE_SIZE_INDIVIDUAL, 
                                               marker_size=MARKER_SIZE, line_width=LINE_WIDTH, 
                                               grid_alpha=GRID_ALPHA, time_margin=LOG_AXIS_MARGIN_FACTOR_TIME,
                                               std_margin=LOG_AXIS_MARGIN_FACTOR_STD, N=N, samples=SAMPLES,
                                               output_dir=OUTPUT_DIR, dpi=DPI):
    """
    Plot angle and tesselation experiments separately in log-log scale and save them.
    """
    print("\n" + "="*60)
    print("PLOTTING INDIVIDUAL EXPERIMENTS (LOG-LOG SCALE)")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot angle noise experiments (log-log)
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        print("\nPlotting angle noise experiments (log-log)...")
        
        plt.figure(figsize=figsize)
        
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    plt.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='o', markersize=marker_size, color=colors[i % len(colors)], 
                              linewidth=line_width, label=f'angle_dev={dev:.3f}')
        
        plt.xlabel('Time Step (log scale)', fontsize=12)
        plt.ylabel('Standard Deviation (log scale)', fontsize=12)
        plt.title('Standard Deviation vs Time - Angle Noise (Log-Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=grid_alpha)
        
        # Set axis limits based on data range for angle noise (log-log)
        all_stds = []
        all_timesteps = []
        for std_values in angle_stds:
            if len(std_values) > 0:
                std_array = np.array(std_values)
                timesteps = np.array(range(1, len(std_values) + 1))
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    all_stds.extend(std_array[valid_mask])
                    all_timesteps.extend(timesteps[valid_mask])
        if all_stds and all_timesteps:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            min_time = np.min(all_timesteps)
            max_time = np.max(all_timesteps)
            plt.xlim(min_time * (1 - time_margin), max_time * (1 + time_margin))
            plt.ylim(min_std * (1 - std_margin), max_std * (1 + std_margin))
        plt.tight_layout()
        
        # Save angle log-log plot
        angle_output = os.path.join(output_dir, f'std_angle_noise_loglog_N{N}_samples{samples}.png')
        plt.savefig(angle_output, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved angle noise log-log plot to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, f'std_angle_noise_loglog_N{N}_samples{samples}.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise log-log plot to {angle_output_pdf}")
        
        plt.close()
        
    else:
        print("\nNo angle noise data to plot.")
    
    # Plot tesselation order experiments (log-log)
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nPlotting tesselation order experiments (log-log)...")
        
        plt.figure(figsize=figsize)
        
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    plt.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='s', markersize=marker_size, color=colors[i % len(colors)], 
                              linewidth=line_width, label=f'shift_prob={prob:.3f}')
        
        plt.xlabel('Time Step (log scale)', fontsize=12)
        plt.ylabel('Standard Deviation (log scale)', fontsize=12)
        plt.title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=grid_alpha)
        
        # Set axis limits based on data range for tesselation noise (log-log)
        all_stds = []
        all_timesteps = []
        for std_values in tesselation_stds:
            if len(std_values) > 0:
                std_array = np.array(std_values)
                timesteps = np.array(range(1, len(std_values) + 1))
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    all_stds.extend(std_array[valid_mask])
                    all_timesteps.extend(timesteps[valid_mask])
        if all_stds and all_timesteps:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            min_time = np.min(all_timesteps)
            max_time = np.max(all_timesteps)
            plt.xlim(min_time * (1 - time_margin), max_time * (1 + time_margin))
            plt.ylim(min_std * (1 - std_margin), max_std * (1 + std_margin))
        
        plt.tight_layout()
        
        # Save tesselation log-log plot
        tess_output = os.path.join(output_dir, f'std_tesselation_order_loglog_N{N}_samples{samples}.png')
        plt.savefig(tess_output, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved tesselation order log-log plot to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, f'std_tesselation_order_loglog_N{N}_samples{samples}.pdf')
        plt.savefig(tess_output_pdf, bbox_inches='tight')
        print(f"✅ Saved tesselation order log-log plot to {tess_output_pdf}")
        
        plt.close()
    else:
        print("\nNo tesselation order data to plot.")

def analyze_and_print_statistics(angle_stds, angle_devs, tesselation_stds, shift_probs):
    """
    Analyze and print statistics about the standard deviation trends.
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        print("\nAngle Noise Analysis:")
        print("-" * 30)
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                final_std = std_values[-1]
                max_std = max(std_values)
                mean_std = np.mean(std_values)
                print(f"  Dev {dev:.3f}: Final STD = {final_std:.2f}, Max STD = {max_std:.2f}, Mean STD = {mean_std:.2f}")
    
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nTesselation Order Noise Analysis:")
        print("-" * 40)
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                final_std = std_values[-1]
                max_std = max(std_values)
                mean_std = np.mean(std_values)
                print(f"  Prob {prob:.3f}: Final STD = {final_std:.2f}, Max STD = {max_std:.2f}, Mean STD = {mean_std:.2f}")

def main():
    """
    Main function to load experiments and create plots.
    Uses global configuration variables defined at the top of the file.
    """
    print("="*60)
    print("ENHANCED STANDARD DEVIATION COMPARISON")
    print("="*60)
    print(f"Using configuration:")
    print(f"  N = {N}")
    print(f"  Steps = {N//4 if DEFAULT_STEPS is None else DEFAULT_STEPS}")
    print(f"  Samples = {SAMPLES}")
    print(f"  Base directory = {BASE_DIR}")
    print(f"  Output directory = {OUTPUT_DIR}")
    print(f"  Angle deviations = {ANGLE_DEVIATIONS}")
    print(f"  Shift probabilities = {SHIFT_PROBABILITIES}")
    
    # Load angle noise experiments
    print("\n" + "-"*40)
    print("LOADING ANGLE NOISE EXPERIMENTS")
    print("-"*40)
    angle_stds, angle_devs = load_and_plot_angle_experiments()
    
    # Load tesselation order experiments
    print("\n" + "-"*40)
    print("LOADING TESSELATION ORDER EXPERIMENTS")
    print("-"*40)
    tesselation_stds, shift_probs = load_and_plot_tesselation_experiments()
    
    # Save data for later analysis
    print("\n" + "-"*40)
    print("SAVING DATA")
    print("-"*40)
    save_plot_data(angle_stds, angle_devs, tesselation_stds, shift_probs, N, SAMPLES)
    
    # Create plots
    print("\n" + "-"*40)
    print("CREATING PLOTS")
    print("-"*40)
    
    # Combined comparison plot
    print("\nCreating combined comparison plot...")
    plot_and_save_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs, N=N, samples=SAMPLES)
    
    # Combined comparison plot (log-log scale)
    print("\nCreating combined comparison plot (log-log scale)...")
    plot_and_save_combined_comparison_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, N=N, samples=SAMPLES)
    
    # Individual plots with save functionality
    plot_individual_experiments_with_save(angle_stds, angle_devs, tesselation_stds, shift_probs, N=N, samples=SAMPLES)
    
    # Individual plots with save functionality (log-log scale)
    plot_individual_experiments_with_save_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, N=N, samples=SAMPLES)
    
    # Statistical analysis
    analyze_and_print_statistics(angle_stds, angle_devs, tesselation_stds, shift_probs)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots and data saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
