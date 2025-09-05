#!/usr/bin/env python3

"""
Simple Deviation vs Saturation Time Visualization

This script creates a clean, focused plot showing the relationship between 
deviation values (Y-axis) and their corresponding saturation times (X-axis)
in the style of plot_linspace_experiment_results.py.

Usage:
    python visualize_deviation_vs_sattime_simple.py
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

# Experiment parameters - EDIT THESE TO MATCH YOUR SETUP
N = 4000                    # System size
theta = math.pi/3           # Theta parameter
base_dir = "experiments_data_samples_linspace_sattime"

# Plot configuration
PLOT_CONFIG = {
    'figure_size': (12, 8),
    'save_figure': True,
    'filename': 'deviation_vs_saturation_time_simple.png',
    'title': 'Deviation vs Saturation Time\nSimple Plateau Detection Method',
    'xlabel': 'Saturation Time',
    'ylabel': 'Deviation Value',
    'fontsize_title': 16,
    'fontsize_labels': 14,
    'fontsize_legend': 12,
    'fontsize_stats': 10,
    'marker_size': 80,
    'marker_color': 'darkblue',
    'marker_alpha': 0.7,
    'marker_edge_color': 'navy',
    'marker_edge_width': 0.5,
    'trend_color': 'red',
    'trend_style': '--',
    'trend_width': 2,
    'trend_alpha': 0.8,
    'grid_alpha': 0.3,
    'dpi': 300,
    'bbox_inches': 'tight'
}

# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_saturation_results(base_dir, N=4000, theta=np.pi/3):
    """
    Collect all saturation time results from the directory structure.
    """
    print(f"=== DEVIATION vs SATURATION TIME VISUALIZATION ===")
    print(f"Collecting saturation results from: {base_dir}")
    print(f"Looking for N={N}, theta={theta:.6f}")
    
    # Format theta for directory name (match the actual directory format)
    theta_folder = f"theta_{theta:.6f}"
    search_dir = os.path.join(base_dir, "static_noise_linspace", f"N_{N}", theta_folder)
    
    print(f"Search directory: {search_dir}")
    
    if not os.path.exists(search_dir):
        print(f"[ERROR] Directory not found: {search_dir}")
        return [], [], [], []
    
    deviations = []
    saturation_times = []
    saturation_values = []
    metadata_list = []
    
    # Find all deviation directories
    dev_dirs = [d for d in os.listdir(search_dir) if d.startswith("dev_min")]
    dev_dirs.sort()
    
    print(f"Found {len(dev_dirs)} deviation directories")
    
    for dev_dir in dev_dirs:
        sattime_file = os.path.join(search_dir, dev_dir, "saturation_time.pkl")
        
        if os.path.exists(sattime_file):
            try:
                with open(sattime_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract deviation value from directory name
                # Format: dev_min0.000_max0.XXX
                if "max" in dev_dir:
                    dev_val = float(dev_dir.split("_max")[1])
                else:
                    dev_val = float(dev_dir.replace("dev_", ""))
                
                sat_time = data.get('saturation_time', np.nan)
                sat_value = data.get('saturation_value', np.nan)
                metadata = data.get('metadata', {})
                
                deviations.append(dev_val)
                saturation_times.append(sat_time)
                saturation_values.append(sat_value)
                metadata_list.append(metadata)
                
                print(f"  {dev_dir}: deviation={dev_val:.3f}, sat_time={sat_time:.2f}, sat_value={sat_value:.4f}")
                
            except Exception as e:
                print(f"  {dev_dir}: ERROR loading data - {e}")
        else:
            print(f"  {dev_dir}: saturation_time.pkl not found")
    
    print(f"\nCollected {len(deviations)} results")
    return deviations, saturation_times, saturation_values, metadata_list

# ============================================================================
# PLOTTING FUNCTION
# ============================================================================

def plot_deviation_vs_saturation_simple(deviations, saturation_times, saturation_values, metadata_list):
    """
    Create a single, clean plot showing deviation vs saturation time.
    """
    config = PLOT_CONFIG
    
    # Convert to numpy arrays and filter valid data
    deviations = np.array(deviations)
    saturation_times = np.array(saturation_times)
    saturation_values = np.array(saturation_values)
    
    # Filter out invalid data points
    valid_mask = (~np.isnan(saturation_times)) & (~np.isnan(saturation_values)) & (saturation_times > 0)
    
    if np.sum(valid_mask) == 0:
        print("[ERROR] No valid data points to plot")
        return {}
    
    valid_devs = deviations[valid_mask]
    valid_sat_times = saturation_times[valid_mask]
    valid_sat_values = saturation_values[valid_mask]
    
    print(f"Plotting {len(valid_devs)} valid results out of {len(deviations)} total")
    
    # Create the plot
    plt.figure(figsize=config['figure_size'])
    
    # Main scatter plot
    plt.scatter(valid_sat_times, valid_devs, 
               s=config['marker_size'], 
               c=config['marker_color'], 
               alpha=config['marker_alpha'],
               edgecolors=config['marker_edge_color'], 
               linewidth=config['marker_edge_width'])
    
    # Add trend line if we have enough points
    if len(valid_devs) > 3:
        try:
            # Fit polynomial (degree 2 for smooth curve)
            degree = min(2, len(valid_devs) - 1)
            coeffs = np.polyfit(valid_sat_times, valid_devs, degree)
            trend_x = np.linspace(valid_sat_times.min(), valid_sat_times.max(), 100)
            trend_y = np.polyval(coeffs, trend_x)
            plt.plot(trend_x, trend_y, 
                    color=config['trend_color'], 
                    linestyle=config['trend_style'], 
                    linewidth=config['trend_width'], 
                    alpha=config['trend_alpha'], 
                    label=f'Trend (degree {degree})')
            plt.legend(fontsize=config['fontsize_legend'])
        except Exception as e:
            print(f"[INFO] Could not fit trend line: {e}")
    
    # Styling
    plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'], fontweight='bold')
    plt.ylabel(config['ylabel'], fontsize=config['fontsize_labels'], fontweight='bold')
    plt.title(config['title'], fontsize=config['fontsize_title'], fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=config['grid_alpha'], linestyle='-', linewidth=0.5)
    
    # Statistical information box
    method_info = "Simple Plateau Detection"
    if metadata_list and len(metadata_list) > 0:
        first_meta = metadata_list[0]
        method = first_meta.get('method', 'Unknown')
        if method == 'plateau_detection':
            window_sizes = first_meta.get('window_sizes', [5, 10, 15])
            threshold = first_meta.get('plateau_threshold', 0.05)
            min_length = first_meta.get('min_plateau_length', 10)
            method_info = f"Plateau Detection\nWindows: {window_sizes}\nThreshold: {threshold*100:.0f}%\nMin length: {min_length}"
    
    stats_text = f"""Data Summary:
• Valid points: {len(valid_devs)} / {len(deviations)}
• Deviation range: {valid_devs.min():.3f} - {valid_devs.max():.3f}
• Saturation time range: {valid_sat_times.min():.1f} - {valid_sat_times.max():.1f}
• Mean sat. time: {valid_sat_times.mean():.1f} ± {valid_sat_times.std():.1f}

Method: {method_info}

System: N={N}, θ={theta:.3f}"""
    
    # Position stats box in upper left
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=config['fontsize_stats'], verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8, edgecolor='navy'))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if config['save_figure']:
        plt.savefig(config['filename'], dpi=config['dpi'], bbox_inches=config['bbox_inches'])
        print(f"\nPlot saved as: {config['filename']}")
    
    # Show the plot
    plt.show()
    
    # Return summary statistics
    return {
        'valid_points': len(valid_devs),
        'total_points': len(deviations),
        'dev_range': (valid_devs.min(), valid_devs.max()),
        'sat_time_range': (valid_sat_times.min(), valid_sat_times.max()),
        'mean_sat_time': valid_sat_times.mean(),
        'std_sat_time': valid_sat_times.std()
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that collects data and creates the plot.
    """
    # Step 1: Collect saturation time data
    deviations, saturation_times, saturation_values, metadata_list = collect_saturation_results(
        base_dir, N, theta
    )
    
    if len(deviations) == 0:
        print("[ERROR] No saturation data found!")
        return
    
    # Step 2: Create the plot
    results = plot_deviation_vs_saturation_simple(
        deviations, saturation_times, saturation_values, metadata_list
    )
    
    # Step 3: Print summary
    if results:
        print(f"\n=== SUMMARY ===")
        print(f"Successfully processed {results['valid_points']} out of {results['total_points']} deviations")
        print(f"Saturation time range: {results['sat_time_range'][0]:.1f} - {results['sat_time_range'][1]:.1f}")
        print(f"Average saturation time: {results['mean_sat_time']:.1f} ± {results['std_sat_time']:.1f}")

if __name__ == "__main__":
    main()
