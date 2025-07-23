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

def save_plot_data(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir="plot_outputs"):
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
            'N': 2000,
            'steps': len(angle_stds[0]) if angle_stds and len(angle_stds[0]) > 0 else 0
        }
        import json
        with open(os.path.join(output_dir, 'angle_noise_std_data.json'), 'w') as f:
            json.dump(angle_data, f, indent=2)
        print(f"✅ Saved angle noise data to {output_dir}/angle_noise_std_data.json")
    
    # Save tesselation data
    if tesselation_stds and shift_probs:
        tesselation_data = {
            'shift_probabilities': shift_probs,
            'std_values': tesselation_stds,
            'timesteps': [list(range(len(std))) for std in tesselation_stds],
            'experiment_type': 'tesselation_order_noise',
            'N': 2000,
            'steps': len(tesselation_stds[0]) if tesselation_stds and len(tesselation_stds[0]) > 0 else 0
        }
        with open(os.path.join(output_dir, 'tesselation_order_std_data.json'), 'w') as f:
            json.dump(tesselation_data, f, indent=2)
        print(f"✅ Saved tesselation order data to {output_dir}/tesselation_order_std_data.json")

def load_and_plot_angle_experiments(N=2000, steps=None, base_dir="experiments_data_samples_probDist"):
    """
    Load angle noise experiments and calculate standard deviations.
    """
    if steps is None:
        steps = N // 4
    
    # Parameters from the angle experiments
    devs = [0, (np.pi/3)/2.5, (np.pi/3) * 2]
    
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

def load_and_plot_tesselation_experiments(N=2000, steps=None, base_dir="experiments_data_samples_probDist"):
    """
    Load tesselation order noise experiments and calculate standard deviations.
    """
    if steps is None:
        steps = N // 4
    
    # Parameters from the tesselation experiments
    shift_probs = [0, 0.2, 0.5]
    
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

def plot_and_save_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir="plot_outputs"):
    """
    Plot both angle and tesselation experiments in a combined figure and save to file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot angle noise results
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        ax1.set_title('Standard Deviation vs Time - Angle Noise', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Standard Deviation', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax1.plot(timesteps, std_values, marker='o', markersize=2, 
                        color=colors[i % len(colors)], linewidth=1.5,
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
                ax1.set_ylim(min_std * 0.9, max_std * 1.1)
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Standard Deviation vs Time - Angle Noise (No Data)')
    
    # Plot tesselation order results
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Standard Deviation', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax2.plot(timesteps, std_values, marker='s', markersize=2, 
                        color=colors[i % len(colors)], linewidth=1.5,
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
                ax2.set_ylim(min_std * 0.9, max_std * 1.1)
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (No Data)')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'std_comparison_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined plot to {output_file}")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, 'std_comparison_combined.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined plot to {output_file_pdf}")
    
    plt.show()

def plot_and_save_combined_comparison_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir="plot_outputs"):
    """
    Plot both angle and tesselation experiments in log-log scale to reveal scaling behavior.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot angle noise results (log-log)
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        ax1.set_title('Standard Deviation vs Time - Angle Noise (Log-Log)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Step (log scale)', fontsize=12)
        ax1.set_ylabel('Standard Deviation (log scale)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    ax1.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='o', markersize=2, color=colors[i % len(colors)], 
                              linewidth=1.5, label=f'angle_dev={dev:.3f}')
        
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
                ax1.set_xlim(min_time * 0.8, max_time * 1.2)
                ax1.set_ylim(min_std * 0.5, max_std * 2)
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Standard Deviation vs Time - Angle Noise (Log-Log, No Data)')
    
    # Plot tesselation order results (log-log)
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step (log scale)', fontsize=12)
        ax2.set_ylabel('Standard Deviation (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    ax2.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='s', markersize=2, color=colors[i % len(colors)], 
                              linewidth=1.5, label=f'shift_prob={prob:.3f}')
        
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
                ax2.set_xlim(min_time * 0.8, max_time * 1.2)
                ax2.set_ylim(min_std * 0.5, max_std * 2)
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log, No Data)')
    
    plt.tight_layout()
    
    # Save the log-log plot
    output_file = os.path.join(output_dir, 'std_comparison_combined_loglog.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined log-log plot to {output_file}")
    
    # Also save as PDF
    output_file_pdf = os.path.join(output_dir, 'std_comparison_combined_loglog.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined log-log plot to {output_file_pdf}")
    
    plt.show()

def plot_individual_experiments_with_save(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir="plot_outputs"):
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
        
        # Use the original plotting function but capture it
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                plt.plot(timesteps, std_values, marker='o', markersize=2, 
                        color=colors[i % len(colors)], linewidth=1.5,
                        label=f'angle_dev={dev:.3f}')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Standard Deviation vs Time for Different Angle Noise Values', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set Y-axis limits based on data range for angle noise
        all_stds = []
        for std_values in angle_stds:
            if len(std_values) > 0:
                all_stds.extend(std_values)
        if all_stds:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            plt.ylim(min_std * 0.9, max_std * 1.1)
        
        plt.tight_layout()
        
        # Save angle plot
        angle_output = os.path.join(output_dir, 'std_angle_noise.png')
        plt.savefig(angle_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved angle noise plot to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, 'std_angle_noise.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise plot to {angle_output_pdf}")
        
        plt.close()
        
    else:
        print("\nNo angle noise data to plot.")
    
    # Plot tesselation order experiments
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nPlotting tesselation order experiments...")
        
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                plt.plot(timesteps, std_values, marker='s', markersize=2, 
                        color=colors[i % len(colors)], linewidth=1.5,
                        label=f'shift_prob={prob:.3f}')
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Standard Deviation', fontsize=12)
        plt.title('Standard Deviation vs Time for Different Tesselation Shift Probabilities', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set Y-axis limits based on data range for tesselation noise
        all_stds = []
        for std_values in tesselation_stds:
            if len(std_values) > 0:
                all_stds.extend(std_values)
        if all_stds:
            min_std = np.min(all_stds)
            max_std = np.max(all_stds)
            plt.ylim(min_std * 0.9, max_std * 1.1)
        
        plt.tight_layout()
        
        # Save tesselation plot
        tess_output = os.path.join(output_dir, 'std_tesselation_order.png')
        plt.savefig(tess_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved tesselation order plot to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, 'std_tesselation_order.pdf')
        plt.savefig(tess_output_pdf, bbox_inches='tight')
        print(f"✅ Saved tesselation order plot to {tess_output_pdf}")
        
        plt.close()
    else:
        print("\nNo tesselation order data to plot.")

def plot_individual_experiments_with_save_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir="plot_outputs"):
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
        
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    plt.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='o', markersize=2, color=colors[i % len(colors)], 
                              linewidth=1.5, label=f'angle_dev={dev:.3f}')
        
        plt.xlabel('Time Step (log scale)', fontsize=12)
        plt.ylabel('Standard Deviation (log scale)', fontsize=12)
        plt.title('Standard Deviation vs Time - Angle Noise (Log-Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
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
            plt.xlim(min_time * 0.8, max_time * 1.2)
            plt.ylim(min_std * 0.5, max_std * 2)
        plt.tight_layout()
        
        # Save angle log-log plot
        angle_output = os.path.join(output_dir, 'std_angle_noise_loglog.png')
        plt.savefig(angle_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved angle noise log-log plot to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, 'std_angle_noise_loglog.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise log-log plot to {angle_output_pdf}")
        
        plt.close()
        
    else:
        print("\nNo angle noise data to plot.")
    
    # Plot tesselation order experiments (log-log)
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nPlotting tesselation order experiments (log-log)...")
        
        plt.figure(figsize=(10, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                # Filter out zero values and start from timestep 1 for log scale
                timesteps = np.array(range(1, len(std_values) + 1))
                std_array = np.array(std_values)
                
                # Only plot positive values (required for log scale)
                valid_mask = std_array > 0
                if np.any(valid_mask):
                    plt.loglog(timesteps[valid_mask], std_array[valid_mask], 
                              marker='s', markersize=2, color=colors[i % len(colors)], 
                              linewidth=1.5, label=f'shift_prob={prob:.3f}')
        
        plt.xlabel('Time Step (log scale)', fontsize=12)
        plt.ylabel('Standard Deviation (log scale)', fontsize=12)
        plt.title('Standard Deviation vs Time - Tesselation Order Noise (Log-Log Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
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
            plt.xlim(min_time * 0.8, max_time * 1.2)
            plt.ylim(min_std * 0.5, max_std * 2)
        
        plt.tight_layout()
        
        # Save tesselation log-log plot
        tess_output = os.path.join(output_dir, 'std_tesselation_order_loglog.png')
        plt.savefig(tess_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved tesselation order log-log plot to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, 'std_tesselation_order_loglog.pdf')
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
    """
    print("="*60)
    print("ENHANCED STANDARD DEVIATION COMPARISON")
    print("="*60)
    
    base_dir = "experiments_data_samples_probDist"
    output_dir = "plot_outputs"
    
    # Load angle noise experiments (N=2000)
    print("\n" + "-"*40)
    print("LOADING ANGLE NOISE EXPERIMENTS")
    print("-"*40)
    angle_stds, angle_devs = load_and_plot_angle_experiments(N=2000, base_dir=base_dir)
    
    # Load tesselation order experiments (N=2000)
    print("\n" + "-"*40)
    print("LOADING TESSELATION ORDER EXPERIMENTS")
    print("-"*40)
    tesselation_stds, shift_probs = load_and_plot_tesselation_experiments(N=2000, base_dir=base_dir)
    
    # Save data for later analysis
    print("\n" + "-"*40)
    print("SAVING DATA")
    print("-"*40)
    save_plot_data(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir)
    
    # Create plots
    print("\n" + "-"*40)
    print("CREATING PLOTS")
    print("-"*40)
    
    # Combined comparison plot
    print("\nCreating combined comparison plot...")
    plot_and_save_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir)
    
    # Combined comparison plot (log-log scale)
    print("\nCreating combined comparison plot (log-log scale)...")
    plot_and_save_combined_comparison_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir)
    
    # Individual plots with save functionality
    plot_individual_experiments_with_save(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir)
    
    # Individual plots with save functionality (log-log scale)
    plot_individual_experiments_with_save_loglog(angle_stds, angle_devs, tesselation_stds, shift_probs, output_dir)
    
    # Statistical analysis
    analyze_and_print_statistics(angle_stds, angle_devs, tesselation_stds, shift_probs)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots and data saved to: {output_dir}/")

if __name__ == "__main__":
    main()
