"""
Final Step Probability Distribution Comparison for Angle and Tesselation Order Noise

This script loads probability distributions from existing experiments and plots
the final probability distributions to compare the spreading effects of different
noise types in the last time step.
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
    prob_distributions2std
)

def load_final_probability_distributions(experiment_type="angle", N=2000, steps=None, base_dir="experiments_data_samples_probDist"):
    """
    Load the final probability distributions from experiments.
    
    Parameters
    ----------
    experiment_type : str
        Either "angle" or "tesselation"
    N : int
        Number of nodes in the graph
    steps : int
        Number of time steps (if None, uses N//4)
    base_dir : str
        Base directory for probability distribution files
        
    Returns
    -------
    tuple
        (final_distributions, parameter_values, parameter_name)
    """
    if steps is None:
        steps = N // 4
    
    if experiment_type == "angle":
        # Parameters from the angle experiments
        parameter_values = [0, (np.pi/3)/2.5, (np.pi/3) * 2]
        parameter_name = "angle_dev"
        noise_type = "angle"
        print(f"Loading angle noise final distributions for N={N}, final step={steps-1}")
        print(f"Angle deviations: {parameter_values}")
    elif experiment_type == "tesselation":
        # Parameters from the tesselation experiments
        parameter_values = [0, 0.2, 0.5]
        parameter_name = "shift_prob"
        noise_type = "tesselation_order"
        print(f"Loading tesselation order final distributions for N={N}, final step={steps-1}")
        print(f"Shift probabilities: {parameter_values}")
    else:
        raise ValueError("experiment_type must be 'angle' or 'tesselation'")
    
    # Check if experiments exist
    if not check_mean_probability_distributions_exist(
        even_line_two_tesselation, N, steps, parameter_values, base_dir, noise_type
    ):
        print(f"❌ {experiment_type.title()} experiments not found in the expected directory.")
        print(f"Expected directory: {base_dir}")
        return None, None, None
    
    # Load the mean probability distributions
    results = load_mean_probability_distributions(
        even_line_two_tesselation, N, steps, parameter_values, base_dir, noise_type
    )
    
    # Extract final distributions (last time step)
    final_distributions = []
    for i, param_prob_dists in enumerate(results):
        if param_prob_dists and len(param_prob_dists) > 0:
            final_dist = param_prob_dists[-1]  # Last time step
            if final_dist is not None:
                final_distributions.append(final_dist)
                print(f"{experiment_type.title()} {i} ({parameter_name}={parameter_values[i]:.3f}): Final distribution loaded")
            else:
                print(f"{experiment_type.title()} {i} ({parameter_name}={parameter_values[i]:.3f}): No final distribution")
                final_distributions.append(None)
        else:
            print(f"{experiment_type.title()} {i} ({parameter_name}={parameter_values[i]:.3f}): No probability distributions")
            final_distributions.append(None)
    
    return final_distributions, parameter_values, parameter_name

def plot_final_distributions_comparison(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000, output_dir="plot_outputs"):
    """
    Plot final probability distributions for both experiment types in comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create domain centered around the middle
    domain = np.arange(N) - N//2
    
    # Create combined comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot angle noise final distributions
    if angle_dists and any(dist is not None for dist in angle_dists):
        ax1.set_title('Final Probability Distributions - Angle Noise', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position (centered)', fontsize=12)
        ax1.set_ylabel('Probability (log scale)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                
                # Normalize to ensure it's a proper probability distribution
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter out zero values for log scale - ensure domain and prob_dist stay aligned
                valid_mask = prob_dist > 0
                if np.any(valid_mask):
                    valid_domain = domain[valid_mask]
                    valid_probs = prob_dist[valid_mask]
                    ax1.semilogy(valid_domain, valid_probs, color=colors[i % len(colors)], linewidth=2,
                            label=f'angle_dev={dev:.3f}', alpha=0.8)
        
        ax1.legend()
        ax1.set_xlim(-N//4, N//4)  # Focus on the central region
        
        # Set Y-axis limits based on data range for angle noise (improved for log scale)
        if angle_dists and any(dist is not None for dist in angle_dists):
            all_probs = []
            for dist in angle_dists:
                if dist is not None:
                    prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                    prob_dist = prob_dist.flatten() / np.sum(prob_dist.flatten())
                    # Focus on central region for better scaling
                    center_idx = len(prob_dist) // 2
                    quarter_range = N // 4
                    central_region = prob_dist[center_idx - quarter_range:center_idx + quarter_range]
                    valid_probs = central_region[central_region > 0]
                    if len(valid_probs) > 0:
                        all_probs.extend(valid_probs)
            if all_probs:
                min_prob = np.min(all_probs)
                max_prob = np.max(all_probs)
                # Use more aggressive bounds for log scale, but ensure min is not too small
                log_min = max(min_prob * 0.01, 1e-10)  # Prevent extremely small values
                log_max = max_prob * 100
                ax1.set_ylim(log_min, log_max)
                print(f"Angle Y-axis range: {log_min:.2e} to {log_max:.2e}")
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Final Probability Distributions - Angle Noise (No Data)')
    
    # Plot tesselation order final distributions
    if tesselation_dists and any(dist is not None for dist in tesselation_dists):
        ax2.set_title('Final Probability Distributions - Tesselation Order Noise', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position (centered)', fontsize=12)
        ax2.set_ylabel('Probability (log scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                
                # Normalize to ensure it's a proper probability distribution
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter out zero values for log scale - ensure domain and prob_dist stay aligned
                valid_mask = prob_dist > 0
                if np.any(valid_mask):
                    valid_domain = domain[valid_mask]
                    valid_probs = prob_dist[valid_mask]
                    ax2.semilogy(valid_domain, valid_probs, color=colors[i % len(colors)], linewidth=2,
                            label=f'shift_prob={prob:.3f}', alpha=0.8)
        
        ax2.legend()
        ax2.set_xlim(-N//4, N//4)  # Focus on the central region
        
        # Set Y-axis limits based on data range for tesselation noise (improved for log scale)
        if tesselation_dists and any(dist is not None for dist in tesselation_dists):
            all_probs = []
            for dist in tesselation_dists:
                if dist is not None:
                    prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                    prob_dist = prob_dist.flatten() / np.sum(prob_dist.flatten())
                    # Focus on central region for better scaling
                    center_idx = len(prob_dist) // 2
                    quarter_range = N // 4
                    central_region = prob_dist[center_idx - quarter_range:center_idx + quarter_range]
                    valid_probs = central_region[central_region > 0]
                    if len(valid_probs) > 0:
                        all_probs.extend(valid_probs)
            if all_probs:
                min_prob = np.min(all_probs)
                max_prob = np.max(all_probs)
                # Use more aggressive bounds for log scale, but ensure min is not too small
                log_min = max(min_prob * 0.01, 1e-10)  # Prevent extremely small values
                log_max = max_prob * 100
                ax2.set_ylim(log_min, log_max)
                print(f"Tesselation Y-axis range: {log_min:.2e} to {log_max:.2e}")
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Final Probability Distributions - Tesselation Order Noise (No Data)')
    
    plt.tight_layout()
    
    # Save the combined plot
    output_file = os.path.join(output_dir, 'final_distributions_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined final distributions plot to {output_file}")
    
    output_file_pdf = os.path.join(output_dir, 'final_distributions_comparison.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined final distributions plot to {output_file_pdf}")
    
    plt.show()

def plot_individual_final_distributions(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000, output_dir="plot_outputs"):
    """
    Plot final probability distributions for each experiment type separately.
    """
    os.makedirs(output_dir, exist_ok=True)
    domain = np.arange(N) - N//2
    
    # Plot angle noise final distributions
    if angle_dists and any(dist is not None for dist in angle_dists):
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter out zero values for log scale - ensure domain and prob_dist stay aligned
                valid_mask = prob_dist > 0
                if np.any(valid_mask):
                    valid_domain = domain[valid_mask]
                    valid_probs = prob_dist[valid_mask]
                    plt.semilogy(valid_domain, valid_probs, color=colors[i % len(colors)], linewidth=2,
                            label=f'angle_dev={dev:.3f}', alpha=0.8)
        
        plt.xlabel('Position (centered)', fontsize=12)
        plt.ylabel('Probability (log scale)', fontsize=12)
        plt.title('Final Probability Distributions - Angle Noise Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-N//4, N//4)
        
        # Set Y-axis limits based on data range for angle noise (improved for log scale)
        all_probs = []
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten() / np.sum(prob_dist.flatten())
                # Focus on central region for better scaling
                center_idx = len(prob_dist) // 2
                quarter_range = N // 4
                central_region = prob_dist[center_idx - quarter_range:center_idx + quarter_range]
                valid_probs = central_region[central_region > 0]
                if len(valid_probs) > 0:
                    all_probs.extend(valid_probs)
        if all_probs:
            min_prob = np.min(all_probs)
            max_prob = np.max(all_probs)
            # Use more aggressive bounds for log scale, but ensure min is not too small
            log_min = max(min_prob * 0.01, 1e-10)  # Prevent extremely small values
            log_max = max_prob * 100
            plt.ylim(log_min, log_max)
            print(f"Individual Angle Y-axis range: {log_min:.2e} to {log_max:.2e}")
        
        plt.tight_layout()
        
        # Save angle plot
        angle_output = os.path.join(output_dir, 'final_distributions_angle_noise.png')
        plt.savefig(angle_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved angle noise final distributions to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, 'final_distributions_angle_noise.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise final distributions to {angle_output_pdf}")
        
        plt.close()
    else:
        print("No angle noise final distributions to plot.")
    
    # Plot tesselation order final distributions
    if tesselation_dists and any(dist is not None for dist in tesselation_dists):
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter out zero values for log scale - ensure domain and prob_dist stay aligned
                valid_mask = prob_dist > 0
                if np.any(valid_mask):
                    valid_domain = domain[valid_mask]
                    valid_probs = prob_dist[valid_mask]
                    plt.semilogy(valid_domain, valid_probs, color=colors[i % len(colors)], linewidth=2,
                            label=f'shift_prob={prob:.3f}', alpha=0.8)
        
        plt.xlabel('Position (centered)', fontsize=12)
        plt.ylabel('Probability (log scale)', fontsize=12)
        plt.title('Final Probability Distributions - Tesselation Order Noise Comparison', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-N//4, N//4)
        
        # Set Y-axis limits based on data range for tesselation noise (improved for log scale)
        all_probs = []
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten() / np.sum(prob_dist.flatten())
                # Focus on central region for better scaling
                center_idx = len(prob_dist) // 2
                quarter_range = N // 4
                central_region = prob_dist[center_idx - quarter_range:center_idx + quarter_range]
                valid_probs = central_region[central_region > 0]
                if len(valid_probs) > 0:
                    all_probs.extend(valid_probs)
        if all_probs:
            min_prob = np.min(all_probs)
            max_prob = np.max(all_probs)
            # Use more aggressive bounds for log scale, but ensure min is not too small
            log_min = max(min_prob * 0.01, 1e-10)  # Prevent extremely small values
            log_max = max_prob * 100
            plt.ylim(log_min, log_max)
            print(f"Individual Tesselation Y-axis range: {log_min:.2e} to {log_max:.2e}")
        
        plt.tight_layout()
        
        # Save tesselation plot
        tess_output = os.path.join(output_dir, 'final_distributions_tesselation_order.png')
        plt.savefig(tess_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved tesselation order final distributions to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, 'final_distributions_tesselation_order.pdf')
        plt.savefig(tess_output_pdf, bbox_inches='tight')
        print(f"✅ Saved tesselation order final distributions to {tess_output_pdf}")
        
        plt.close()
    else:
        print("No tesselation order final distributions to plot.")

def plot_final_distributions_comparison_loglinear(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000, output_dir="plot_outputs"):
    """
    Plot final probability distributions for both experiment types in comparison plots with log X linear scale.
    Uses logarithmic scale on X-axis (position) and linear scale on Y-axis (probability).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create domain centered around the middle
    domain = np.arange(N) - N//2
    
    # Create combined comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot angle noise final distributions (log-linear)
    if angle_dists and any(dist is not None for dist in angle_dists):
        ax1.set_title('Final Probability Distributions - Angle Noise (Log-Linear)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position (log scale)', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                
                # Normalize to ensure it's a proper probability distribution
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter for positive positions only (required for log scale)
                positive_mask = domain > 0
                if np.any(positive_mask):
                    ax1.semilogx(domain[positive_mask], prob_dist[positive_mask], 
                                color=colors[i % len(colors)], linewidth=2,
                                label=f'angle_dev={dev:.3f}', alpha=0.8)
        
        ax1.legend()
        ax1.set_xlim(1, N//4)  # Focus on positive positions for log scale
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Final Probability Distributions - Angle Noise (No Data)')
    
    # Plot tesselation order final distributions (log-linear)
    if tesselation_dists and any(dist is not None for dist in tesselation_dists):
        ax2.set_title('Final Probability Distributions - Tesselation Order Noise (Log-Linear)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Position (log scale)', fontsize=12)
        ax2.set_ylabel('Probability', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                
                # Normalize to ensure it's a proper probability distribution
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter for positive positions only (required for log scale)
                positive_mask = domain > 0
                if np.any(positive_mask):
                    ax2.semilogx(domain[positive_mask], prob_dist[positive_mask], 
                                color=colors[i % len(colors)], linewidth=2,
                                label=f'shift_prob={prob:.3f}', alpha=0.8)
        
        ax2.legend()
        ax2.set_xlim(1, N//4)  # Focus on positive positions for log scale
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Final Probability Distributions - Tesselation Order Noise (No Data)')
    
    plt.tight_layout()
    
    # Save the combined log-linear plot
    output_file = os.path.join(output_dir, 'final_distributions_comparison_loglinear.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved combined final distributions log-linear plot to {output_file}")
    
    output_file_pdf = os.path.join(output_dir, 'final_distributions_comparison_loglinear.pdf')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved combined final distributions log-linear plot to {output_file_pdf}")
    
    plt.close()  # Close instead of show to avoid interruption

def plot_individual_final_distributions_loglinear(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000, output_dir="plot_outputs"):
    """
    Plot final probability distributions for each experiment type separately with log X linear scale.
    """
    os.makedirs(output_dir, exist_ok=True)
    domain = np.arange(N) - N//2
    
    # Plot angle noise final distributions (log-linear)
    if angle_dists and any(dist is not None for dist in angle_dists):
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter for positive positions only (required for log scale)
                positive_mask = domain > 0
                if np.any(positive_mask):
                    plt.semilogx(domain[positive_mask], prob_dist[positive_mask], 
                                color=colors[i % len(colors)], linewidth=2,
                                label=f'angle_dev={dev:.3f}', alpha=0.8)
        
        plt.xlabel('Position (log scale)', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Final Probability Distributions - Angle Noise (Log-Linear Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, N//4)
        plt.tight_layout()
        
        # Save angle log-linear plot
        angle_output = os.path.join(output_dir, 'final_distributions_angle_noise_loglinear.png')
        plt.savefig(angle_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved angle noise final distributions log-linear plot to {angle_output}")
        
        angle_output_pdf = os.path.join(output_dir, 'final_distributions_angle_noise_loglinear.pdf')
        plt.savefig(angle_output_pdf, bbox_inches='tight')
        print(f"✅ Saved angle noise final distributions log-linear plot to {angle_output_pdf}")
        
        plt.close()
    else:
        print("No angle noise final distributions to plot (log-linear).")
    
    # Plot tesselation order final distributions (log-linear)
    if tesselation_dists and any(dist is not None for dist in tesselation_dists):
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Filter for positive positions only (required for log scale)
                positive_mask = domain > 0
                if np.any(positive_mask):
                    plt.semilogx(domain[positive_mask], prob_dist[positive_mask], 
                                color=colors[i % len(colors)], linewidth=2,
                                label=f'shift_prob={prob:.3f}', alpha=0.8)
        
        plt.xlabel('Position (log scale)', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Final Probability Distributions - Tesselation Order Noise (Log-Linear Scale)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, N//4)
        plt.tight_layout()
        
        # Save tesselation log-linear plot
        tess_output = os.path.join(output_dir, 'final_distributions_tesselation_order_loglinear.png')
        plt.savefig(tess_output, dpi=300, bbox_inches='tight')
        print(f"✅ Saved tesselation order final distributions log-linear plot to {tess_output}")
        
        tess_output_pdf = os.path.join(output_dir, 'final_distributions_tesselation_order_loglinear.pdf')
        plt.savefig(tess_output_pdf, bbox_inches='tight')
        print(f"✅ Saved tesselation order final distributions log-linear plot to {tess_output_pdf}")
        
        plt.close()
    else:
        print("No tesselation order final distributions to plot (log-linear).")

def analyze_final_distributions(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000):
    """
    Analyze and print statistics about the final probability distributions.
    """
    print("\n" + "="*60)
    print("FINAL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    domain = np.arange(N) - N//2
    
    if angle_dists and any(dist is not None for dist in angle_dists):
        print("\nAngle Noise Final Distribution Analysis:")
        print("-" * 45)
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Calculate statistics
                mean_pos = np.sum(domain * prob_dist)
                variance = np.sum((domain - mean_pos)**2 * prob_dist)
                std_dev = np.sqrt(variance)
                max_prob = np.max(prob_dist)
                max_pos = domain[np.argmax(prob_dist)]
                
                print(f"  Dev {dev:.3f}: Mean={mean_pos:.2f}, Std={std_dev:.2f}, Max_prob={max_prob:.6f} at pos={max_pos}")
    
    if tesselation_dists and any(dist is not None for dist in tesselation_dists):
        print("\nTesselation Order Noise Final Distribution Analysis:")
        print("-" * 55)
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                
                # Calculate statistics
                mean_pos = np.sum(domain * prob_dist)
                variance = np.sum((domain - mean_pos)**2 * prob_dist)
                std_dev = np.sqrt(variance)
                max_prob = np.max(prob_dist)
                max_pos = domain[np.argmax(prob_dist)]
                
                print(f"  Prob {prob:.3f}: Mean={mean_pos:.2f}, Std={std_dev:.2f}, Max_prob={max_prob:.6f} at pos={max_pos}")

def save_final_distribution_data(angle_dists, angle_devs, tesselation_dists, shift_probs, N=2000, output_dir="plot_outputs"):
    """
    Save the final distribution data to files for later analysis.
    """
    os.makedirs(output_dir, exist_ok=True)
    domain = np.arange(N) - N//2
    
    # Save angle data
    if angle_dists and angle_devs:
        angle_data = {
            'deviations': angle_devs,
            'domain': domain.tolist(),
            'final_distributions': [],
            'experiment_type': 'angle_noise_final',
            'N': N,
            'description': 'Final probability distributions for angle noise experiments'
        }
        
        for i, (dist, dev) in enumerate(zip(angle_dists, angle_devs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                angle_data['final_distributions'].append(prob_dist.tolist())
            else:
                angle_data['final_distributions'].append(None)
        
        import json
        with open(os.path.join(output_dir, 'final_distributions_angle_data.json'), 'w') as f:
            json.dump(angle_data, f, indent=2)
        print(f"✅ Saved angle final distribution data to {output_dir}/final_distributions_angle_data.json")
    
    # Save tesselation data
    if tesselation_dists and shift_probs:
        tesselation_data = {
            'shift_probabilities': shift_probs,
            'domain': domain.tolist(),
            'final_distributions': [],
            'experiment_type': 'tesselation_order_noise_final',
            'N': N,
            'description': 'Final probability distributions for tesselation order noise experiments'
        }
        
        for i, (dist, prob) in enumerate(zip(tesselation_dists, shift_probs)):
            if dist is not None:
                prob_dist = np.abs(dist)**2 if np.iscomplexobj(dist) else dist
                prob_dist = prob_dist.flatten()
                prob_dist = prob_dist / np.sum(prob_dist)
                tesselation_data['final_distributions'].append(prob_dist.tolist())
            else:
                tesselation_data['final_distributions'].append(None)
        
        with open(os.path.join(output_dir, 'final_distributions_tesselation_data.json'), 'w') as f:
            json.dump(tesselation_data, f, indent=2)
        print(f"✅ Saved tesselation final distribution data to {output_dir}/final_distributions_tesselation_data.json")

def main():
    """
    Main function to load experiments and create final distribution plots.
    """
    print("="*60)
    print("FINAL PROBABILITY DISTRIBUTION COMPARISON")
    print("="*60)
    
    base_dir = "experiments_data_samples_probDist"
    output_dir = "plot_outputs"
    N = 2000
    
    # Load angle noise final distributions
    print("\n" + "-"*40)
    print("LOADING ANGLE NOISE FINAL DISTRIBUTIONS")
    print("-"*40)
    angle_dists, angle_devs, _ = load_final_probability_distributions("angle", N=N, base_dir=base_dir)
    
    # Load tesselation order final distributions
    print("\n" + "-"*40)
    print("LOADING TESSELATION ORDER FINAL DISTRIBUTIONS")
    print("-"*40)
    tesselation_dists, shift_probs, _ = load_final_probability_distributions("tesselation", N=N, base_dir=base_dir)
    
    # Save data for later analysis
    print("\n" + "-"*40)
    print("SAVING FINAL DISTRIBUTION DATA")
    print("-"*40)
    save_final_distribution_data(angle_dists, angle_devs, tesselation_dists, shift_probs, N, output_dir)
    
    # Create plots
    print("\n" + "-"*40)
    print("CREATING FINAL DISTRIBUTION PLOTS")
    print("-"*40)
    
    # Combined comparison plot
    print("\nCreating combined final distribution comparison plot...")
    plot_final_distributions_comparison(angle_dists, angle_devs, tesselation_dists, shift_probs, N, output_dir)
    
    # Combined comparison plot (log-linear)
    print("\nCreating combined final distribution comparison plot (log-linear)...")
    plot_final_distributions_comparison_loglinear(angle_dists, angle_devs, tesselation_dists, shift_probs, N, output_dir)
    
    # Individual plots
    print("\nCreating individual final distribution plots...")
    plot_individual_final_distributions(angle_dists, angle_devs, tesselation_dists, shift_probs, N, output_dir)
    
    # Individual plots (log-linear)
    print("\nCreating individual final distribution plots (log-linear)...")
    plot_individual_final_distributions_loglinear(angle_dists, angle_devs, tesselation_dists, shift_probs, N, output_dir)
    
    # Statistical analysis
    analyze_final_distributions(angle_dists, angle_devs, tesselation_dists, shift_probs, N)
    
    print("\n" + "="*60)
    print("FINAL DISTRIBUTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots and data saved to: {output_dir}/")

if __name__ == "__main__":
    main()
