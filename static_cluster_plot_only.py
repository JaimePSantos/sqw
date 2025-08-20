#!/usr/bin/env python3

"""
Static Noise Experiment - Plotting Only Module (Optimized)

This script loads existing probability distribution and standard deviation data
and creates the plots defined in the main static cluster experiment.

OPTIMIZATIONS:
- Only loads final step probability distributions for final distribution plotting
- Directly loads pre-computed standard deviation data from files
- Significantly faster than loading all time steps for plotting

It extracts all necessary data loading functions and plotting routines
from static_cluster_logged_mp.py to provide a standalone plotting capability.

Usage:
    python static_cluster_plot_only.py
    
Configuration:
- Modify the plotting parameters and deviation values below
- Ensure the experiments_data folders exist with the required data
"""

import time
import math
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Global plotting control
ENABLE_PLOTTING = True  # Master switch for all plotting

# Standard Deviation vs Time Plot Configuration
STD_PLOT_CONFIG = {
    'enabled': True,                    # Enable/disable this specific plot
    'use_loglog': True,                 # Use log-log scale instead of linear
    'figure_size': (12, 8),            # Figure size in inches (width, height)
    'save_figure': True,                # Save plot to file
    'filename_linear': 'static_noise_std_vs_time.png',
    'filename_loglog': 'static_noise_std_vs_time_loglog.png',
    'title_linear': 'Standard Deviation vs Time for Different Static Noise Deviations',
    'title_loglog': 'Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations',
    'xlabel': 'Time Step',
    'ylabel': 'Standard Deviation',
    'fontsize_title': 14,
    'fontsize_labels': 12,
    'fontsize_legend': 10,
    'linewidth': 2,
    'markersize': 0,
    'marker_circle': 'o',
    'marker_square': 's',
    'marker_square_size': 4,
    'linestyle_noiseless': '--',
    'alpha_noiseless': 0.8,
    'alpha_regular': 1.0,
    'grid_alpha': 0.3,
    'grid_which': 'both',               # For log-log: 'both', 'major', 'minor'
    'grid_linestyle': '-',
    'epsilon_noiseless': 1e-3,          # Small value for noiseless case in log-log
    'dpi': 300,                         # DPI for saved figures
    'bbox_inches': 'tight'              # Bbox setting for saved figures
}

# Final Probability Distribution Plot Configuration  
PROBDIST_PLOT_CONFIG = {
    'enabled': True,                    # Enable/disable this specific plot
    'figure_size': (14, 8),            # Figure size in inches (width, height)
    'save_figure': True,                # Save plot to file
    'filename': 'static_noise_final_probdist_log.png',
    'title': 'Probability Distributions at Final Time Step (t={final_step}) - Log Scale',
    'xlabel': 'Position',
    'ylabel': 'Probability (log scale)',
    'fontsize_title': 14,
    'fontsize_labels': 12,
    'fontsize_legend': 10,
    'linewidth': 2,
    'alpha': 0.8,
    'xlim': (-150, 150),               # X-axis limits
    'ylim_min': 1e-5,                  # Y-axis minimum (None for auto)
    'ylim_max': 1e-1,                  # Y-axis maximum (None for auto)
    'grid_alpha': 0.3,
    'dpi': 300,                        # DPI for saved figures
    'bbox_inches': 'tight'             # Bbox setting for saved figures
}

# Label formatting configuration
LABEL_CONFIG = {
    'precision_new_format': 3,          # Decimal places for new format labels
    'precision_legacy_format': 3,       # Decimal places for legacy format labels
    'precision_single_value': 3,        # Decimal places for single value labels
    'new_format_template': 'max{max_dev:.{prec}f}_min{min_val:.{prec}f}',
    'legacy_format_template': 'min{min_val:.{prec}f}_max{max_val:.{prec}f}',
    'single_value_template': '{dev:.{prec}f}',
    'plot_label_template': 'Static deviation = {dev_label}'
}

# Experiment parameters (must match the original experiment)
N = 20000  # System size
steps = N//4  # Time steps
samples = 10  # Samples per deviation

# Quantum walk parameters
theta = math.pi/3  # Base theta parameter for static noise
initial_state_kwargs = {"nodes": [N//2]}

# Deviation values for static noise experiments
# MUST MATCH the original experiment configuration
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),           # Medium noise range  
]

# Data directories
probdist_base_dir = "experiments_data_samples_probDist"
std_base_dir = "experiments_data_samples_std"
noise_type = "static_noise"

# ============================================================================
# LOADING FUNCTIONS (extracted from smart_loading_static.py)
# ============================================================================

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

def format_deviation_for_filename(deviation_value, use_legacy_format=None):
    """
    Format deviation values for filename use.
    Handles both single values and tuple ranges.
    """
    if isinstance(deviation_value, (tuple, list)) and len(deviation_value) == 2:
        # Tuple format (min, max)
        min_val, max_val = deviation_value
        if use_legacy_format is False:
            # New format: max_dev and min_factor
            return f"max{max_val:.6f}_min{min_val:.6f}"
        else:
            # Legacy format: min and max
            return f"min{min_val:.6f}_max{max_val:.6f}"
    else:
        # Single value format
        return f"{deviation_value:.6f}"

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data", theta=None):
    """
    Construct experiment directory path following the expected structure.
    """
    tesselation_name = tesselation_func.__name__
    
    if noise_type == "static_noise":
        folder = f"{tesselation_name}_static_noise"
        base = os.path.join(base_dir, folder)
        
        if theta is not None:
            theta_folder = f"theta_{theta:.6f}"
            base = os.path.join(base, theta_folder)
        
        if has_noise and noise_params is not None:
            dev_folder = f"dev_{format_deviation_for_filename(noise_params[0])}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    else:
        # Other noise types
        noise_str = f"{noise_type}" if has_noise else f"{noise_type}_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        
        if theta is not None:
            theta_folder = f"theta_{theta:.6f}"
            base = os.path.join(base, theta_folder)
        
        if has_noise and noise_params is not None:
            dev_folder = f"dev_{format_deviation_for_filename(noise_params[0])}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")

def find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data", theta=None):
    """
    Find experiment directory supporting both old and new formats.
    Returns: Tuple (directory_path, found_format)
    """
    if noise_type != "static_noise" or not noise_params:
        return get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta), 'unified'
    
    deviation_range = noise_params[0] if noise_params else 0
    tesselation_name = tesselation_func.__name__
    
    # Try legacy format first (since that's what we have)
    legacy_folder = f"{tesselation_name}_static_noise"
    legacy_base = os.path.join(base_dir, legacy_folder)
    
    if theta is not None:
        theta_folder = f"theta_{theta:.6f}"
        legacy_base = os.path.join(legacy_base, theta_folder)
    
    # For legacy format, use the old naming convention
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        min_val, max_val = deviation_range
        legacy_dev_suffix = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        legacy_dev_suffix = f"{deviation_range:.6f}"
    
    legacy_dev_folder = f"dev_{legacy_dev_suffix}"
    legacy_path = os.path.join(legacy_base, legacy_dev_folder, f"N_{N}")
    
    if os.path.exists(legacy_path):
        return legacy_path, 'legacy'
    
    # Try unified structure as fallback
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        unified_dev_suffix = format_deviation_for_filename(deviation_range, use_legacy_format=False)
    else:
        unified_dev_suffix = format_deviation_for_filename(deviation_range, use_legacy_format=True)
    
    unified_dev_folder = f"dev_{unified_dev_suffix}"
    unified_path = os.path.join(legacy_base, unified_dev_folder, f"N_{N}")
    
    if os.path.exists(unified_path):
        return unified_path, 'unified'
    
    # Return legacy path even if it doesn't exist (for consistency)
    return legacy_path, 'legacy'

def load_final_step_probability_distributions(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type, theta):
    """
    Load only the final step probability distributions from the probDist folder.
    Optimized for final distribution plotting.
    """
    final_step = steps - 1
    print(f"Loading final step probability distributions (step {final_step}) for {len(parameter_list)} deviations...")
    
    results = []
    for i, params in enumerate(parameter_list):
        exp_dir, format_used = find_experiment_dir_flexible(
            tesselation_func, True, N, [params], noise_type, probdist_base_dir, theta
        )
        
        # Files are directly in the experiment directory (not in a probDist subfolder)
        if not os.path.exists(exp_dir):
            print(f"  [MISSING] Experiment directory not found: {exp_dir}")
            results.append(None)
            continue
        
        print(f"  [LOADING] Dev {i+1}/{len(parameter_list)}: {params} from {exp_dir}")
        
        try:
            # Load only the final step
            prob_file_new = os.path.join(exp_dir, f"mean_step_{final_step}.pkl")
            prob_file_old = os.path.join(exp_dir, f"probDist_step_{final_step}.pkl")
            
            prob_file = prob_file_new if os.path.exists(prob_file_new) else prob_file_old
            
            if os.path.exists(prob_file):
                with open(prob_file, 'rb') as f:
                    prob_dist = pickle.load(f)
                    results.append(prob_dist)
                    print(f"  [OK] Loaded final step probability distribution")
            else:
                print(f"  [MISSING] Final step file not found: {prob_file}")
                results.append(None)
            
        except Exception as e:
            print(f"  [ERROR] Failed to load dev {params}: {e}")
            results.append(None)
    
    return results

def load_std_data_directly(devs, N, tesselation_func, std_base_dir, noise_type, theta=None):
    """
    Load standard deviation data directly from std files.
    Optimized for standard deviation plotting.
    """
    print(f"\n[DATA] Loading standard deviation data directly from '{std_base_dir}'...")
    
    stds = []
    
    for i, dev in enumerate(devs):
        print(f"\n[STD {i+1}/{len(devs)}] Loading deviation: {dev}")
        
        # Get experiment directory for this deviation
        exp_dir, format_used = find_experiment_dir_flexible(
            tesselation_func, True, N, [dev], noise_type, std_base_dir, theta
        )
        
        std_file = os.path.join(exp_dir, "std_data.pkl")
        
        # Try to load existing std data
        try:
            if os.path.exists(std_file):
                with open(std_file, 'rb') as f:
                    std_values = pickle.load(f)
                print(f"  [OK] Loaded {len(std_values)} std values from file")
                stds.append(std_values)
            else:
                print(f"  [MISSING] Standard deviation file not found: {std_file}")
                stds.append([])
        except Exception as e:
            print(f"  [WARNING] Could not load std data: {e}")
            stds.append([])
    
    print(f"[OK] Standard deviation data loading completed!")
    return stds

# ============================================================================
# STANDARD DEVIATION LOADING FUNCTIONS
# ============================================================================

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_standard_deviation_vs_time(stds, devs, steps):
    """
    Plot standard deviation vs time for different noise deviations.
    """
    config = STD_PLOT_CONFIG
    if not ENABLE_PLOTTING or not config['enabled']:
        print("\n[PLOT] Standard deviation plotting disabled")
        return
    
    print("\n[PLOT] Creating standard deviation vs time plot...")
    try:
        if len(stds) > 0 and any(len(std) > 0 for std in stds):
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=config['figure_size'])
            
            for i, (std_values, dev) in enumerate(zip(stds, devs)):
                if len(std_values) > 0:
                    time_steps = list(range(len(std_values)))
                    
                    # Format dev for display using label config
                    if isinstance(dev, tuple) and len(dev) == 2:
                        if dev[1] <= 1.0 and dev[1] >= 0.0:
                            # New format: (max_dev, min_factor)
                            max_dev, min_factor = dev
                            dev_label = LABEL_CONFIG['new_format_template'].format(
                                max_dev=max_dev, min_val=max_dev*min_factor, 
                                prec=LABEL_CONFIG['precision_new_format'])
                        else:
                            # Legacy format: (min, max)
                            min_val, max_val = dev
                            dev_label = LABEL_CONFIG['legacy_format_template'].format(
                                min_val=min_val, max_val=max_val, 
                                prec=LABEL_CONFIG['precision_legacy_format'])
                    else:
                        # Single value format
                        dev_label = LABEL_CONFIG['single_value_template'].format(
                            dev=dev, prec=LABEL_CONFIG['precision_single_value'])
                    
                    plot_label = LABEL_CONFIG['plot_label_template'].format(dev_label=dev_label)
                    
                    # Handle zero values for log-log plot
                    if config['use_loglog']:
                        # Check if this is a zero standard deviation case (noiseless)
                        if all(s == 0 for s in std_values):
                            # For noiseless case (std = 0), plot at bottom of y-axis
                            filtered_times = [t for t in time_steps if t > 0]
                            filtered_stds = [config['epsilon_noiseless']] * len(filtered_times)
                            plt.loglog(filtered_times, filtered_stds, 
                                     label=f'{plot_label} (noiseless)', 
                                     marker=config['marker_square'], markersize=config['marker_square_size'], 
                                     linewidth=config['linewidth'], 
                                     linestyle=config['linestyle_noiseless'], alpha=config['alpha_noiseless'])
                        else:
                            # Remove zero values which can't be plotted on log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                            if filtered_data:
                                filtered_times, filtered_stds = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=plot_label, 
                                         marker=config['marker_circle'], markersize=config['markersize'], 
                                         linewidth=config['linewidth'], alpha=config['alpha_regular'])
                    else:
                        plt.plot(time_steps, std_values, 
                               label=plot_label, 
                               marker=config['marker_circle'], markersize=config['markersize'], 
                               linewidth=config['linewidth'], alpha=config['alpha_regular'])
            
            plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
            plt.ylabel(config['ylabel'], fontsize=config['fontsize_labels'])
            
            if config['use_loglog']:
                plt.title(config['title_loglog'], fontsize=config['fontsize_title'])
                plt.grid(True, alpha=config['grid_alpha'], which=config['grid_which'], 
                        linestyle=config['grid_linestyle'])
                plot_filename = config['filename_loglog']
            else:
                plt.title(config['title_linear'], fontsize=config['fontsize_title'])
                plt.grid(True, alpha=config['grid_alpha'])
                plot_filename = config['filename_linear']
            
            plt.legend(fontsize=config['fontsize_legend'])
            plt.tight_layout()
            
            # Save the plot (if enabled)
            if config['save_figure']:
                plt.savefig(plot_filename, dpi=config['dpi'], bbox_inches=config['bbox_inches'])
                print(f"[OK] Plot saved as '{plot_filename}'")
            
            # Show the plot
            plt.show()
            plot_type = "log-log" if config['use_loglog'] else "linear"
            saved_status = " and saved" if config['save_figure'] else ""
            print(f"[OK] Standard deviation plot displayed{saved_status}! (Scale: {plot_type})")
        else:
            print("[WARNING] No standard deviation data available for plotting")
    except Exception as e:
        print(f"[WARNING] Could not create plot: {e}")
        import traceback
        traceback.print_exc()

def plot_final_probability_distributions(final_results, devs, steps, N):
    """
    Plot final probability distributions for different noise deviations.
    final_results: List of final step probability distributions (one per deviation)
    """
    config = PROBDIST_PLOT_CONFIG
    if not ENABLE_PLOTTING or not config['enabled']:
        if not ENABLE_PLOTTING:
            print("\n[PLOT] Plotting disabled (ENABLE_PLOTTING=False)")
        else:
            print("\n[PLOT] Final probability distribution plotting disabled")
        return
    
    print("\n[PLOT] Creating final probability distribution plot...")
    try:
        if final_results and len(final_results) > 0:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=config['figure_size'])
            
            # Use the last time step (steps-1)
            final_step = steps - 1
            domain = np.arange(N) - N//2  # Center domain around 0
            
            for i, (final_prob_dist, dev) in enumerate(zip(final_results, devs)):
                if final_prob_dist is not None:
                    final_prob_dist = final_prob_dist.flatten()
                    
                    # Format dev for display using label config
                    if isinstance(dev, tuple) and len(dev) == 2:
                        if dev[1] <= 1.0 and dev[1] >= 0.0:
                            # New format: (max_dev, min_factor)
                            max_dev, min_factor = dev
                            dev_label = LABEL_CONFIG['new_format_template'].format(
                                max_dev=max_dev, min_val=max_dev*min_factor, 
                                prec=LABEL_CONFIG['precision_new_format'])
                        else:
                            # Legacy format: (min, max)
                            min_val, max_val = dev
                            dev_label = LABEL_CONFIG['legacy_format_template'].format(
                                min_val=min_val, max_val=max_val, 
                                prec=LABEL_CONFIG['precision_legacy_format'])
                    else:
                        # Single value format
                        dev_label = LABEL_CONFIG['single_value_template'].format(
                            dev=dev, prec=LABEL_CONFIG['precision_single_value'])
                    
                    plot_label = LABEL_CONFIG['plot_label_template'].format(dev_label=dev_label)
                    
                    # Plot the probability distribution with log y-axis
                    plt.semilogy(domain, final_prob_dist, 
                               label=plot_label, 
                               linewidth=config['linewidth'], alpha=config['alpha'])
            
            plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
            plt.ylabel(config['ylabel'], fontsize=config['fontsize_labels'])
            plt.title(config['title'].format(final_step=final_step), fontsize=config['fontsize_title'])
            
            # Set axis limits based on config
            if config['xlim']:
                plt.xlim(config['xlim'])
            if config['ylim_min'] is not None or config['ylim_max'] is not None:
                plt.ylim(config['ylim_min'], config['ylim_max'])
            
            plt.legend(fontsize=config['fontsize_legend'])
            plt.grid(True, alpha=config['grid_alpha'])
            plt.tight_layout()
            
            # Save the plot (if enabled)
            if config['save_figure']:
                plt.savefig(config['filename'], dpi=config['dpi'], bbox_inches=config['bbox_inches'])
                print(f"[OK] Probability distribution plot saved as '{config['filename']}'")
            
            # Show the plot
            plt.show()
            saved_status = " and saved" if config['save_figure'] else ""
            print(f"[OK] Final probability distribution plot displayed{saved_status}!")
        else:
            print("[WARNING] No mean probability distribution data available for plotting")
    except Exception as e:
        print(f"[WARNING] Could not create probability distribution plot: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that loads data and creates plots.
    """
    print("="*80)
    print("STATIC NOISE EXPERIMENT - PLOTTING ONLY")
    print("="*80)
    print(f"System size: N = {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Deviation values: {devs}")
    print(f"Theta parameter: {theta}")
    print("")
    
    print("Configuration:")
    print(f"  ENABLE_PLOTTING = {ENABLE_PLOTTING}")
    print(f"  STD_PLOT enabled = {STD_PLOT_CONFIG['enabled']}")
    print(f"  STD_PLOT use_loglog = {STD_PLOT_CONFIG['use_loglog']}")
    print(f"  PROBDIST_PLOT enabled = {PROBDIST_PLOT_CONFIG['enabled']}")
    print(f"  Save figures = {STD_PLOT_CONFIG['save_figure'] and PROBDIST_PLOT_CONFIG['save_figure']}")
    print("")
    
    # Step 1: Load standard deviation data directly
    print("[STEP 1] Loading standard deviation data...")
    try:
        stds = load_std_data_directly(devs, N, dummy_tesselation_func, std_base_dir, noise_type, theta)
        print(f"[OK] Standard deviation data ready for {len([s for s in stds if len(s) > 0])} / {len(devs)} deviations")
    except Exception as e:
        print(f"[ERROR] Failed to load standard deviation data: {e}")
        stds = []
    
    # Step 2: Load final step probability distributions (only if needed for plotting)
    final_results = []
    if PROBDIST_PLOT_CONFIG['enabled']:
        print("[STEP 2] Loading final step probability distributions...")
        try:
            final_results = load_final_step_probability_distributions(
                dummy_tesselation_func, N, steps, devs, probdist_base_dir, noise_type, theta
            )
            print(f"[OK] Loaded final step distributions for {len([r for r in final_results if r is not None])} / {len(devs)} deviations")
        except Exception as e:
            print(f"[ERROR] Failed to load final step probability distributions: {e}")
            final_results = []
    else:
        print("[STEP 2] Skipping final step probability distributions (PLOT_FINAL_PROBDIST=False)")
    
    # Step 3: Create plots
    print("[STEP 3] Creating plots...")
    
    # Plot 1: Standard deviation vs time
    plot_standard_deviation_vs_time(stds, devs, steps)
    
    # Plot 2: Final probability distributions
    plot_final_probability_distributions(final_results, devs, steps, N)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
