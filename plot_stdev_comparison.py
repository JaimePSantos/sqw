"""
Plot Standard Deviation Comparison for Angle and Tesselation Order Noise

This script loads probability distributions from existing experiments and plots
the standard deviation as a function of time steps for both angle noise and
tesselation order noise experiments.
"""

from sqw.tesselations import even_line_two_tesselation
from sqw.states import uniform_initial_state
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from jaime_scripts import (
    load_mean_probability_distributions,
    check_mean_probability_distributions_exist,
    prob_distributions2std,
    plot_std_vs_time_qwak
)

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

def plot_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs):
    """
    Plot both angle and tesselation experiments in a combined figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot angle noise results
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        ax1.set_title('Standard Deviation vs Time - Angle Noise')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Standard Deviation')
        ax1.grid(True)
        
        for i, (std_values, dev) in enumerate(zip(angle_stds, angle_devs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax1.plot(timesteps, std_values, marker='o', markersize=3, label=f'angle_dev={dev:.3f}')
        
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No angle noise data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title('Standard Deviation vs Time - Angle Noise (No Data)')
    
    # Plot tesselation order results
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True)
        
        for i, (std_values, prob) in enumerate(zip(tesselation_stds, shift_probs)):
            if len(std_values) > 0:
                timesteps = list(range(len(std_values)))
                ax2.plot(timesteps, std_values, marker='s', markersize=3, label=f'shift_prob={prob:.3f}')
        
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No tesselation order data available', 
                horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax2.set_title('Standard Deviation vs Time - Tesselation Order Noise (No Data)')
    
    plt.tight_layout()
    plt.show()

def plot_individual_experiments(angle_stds, angle_devs, tesselation_stds, shift_probs):
    """
    Plot angle and tesselation experiments separately using the original plotting function.
    """
    print("\n" + "="*60)
    print("PLOTTING INDIVIDUAL EXPERIMENTS")
    print("="*60)
    
    # Plot angle noise experiments
    if angle_stds and any(len(std) > 0 for std in angle_stds):
        print("\nPlotting angle noise experiments...")
        plot_std_vs_time_qwak(
            angle_stds, 
            angle_devs, 
            title_prefix="Angle noise (mean)", 
            parameter_name="dev"
        )
    else:
        print("\nNo angle noise data to plot.")
    
    # Plot tesselation order experiments
    if tesselation_stds and any(len(std) > 0 for std in tesselation_stds):
        print("\nPlotting tesselation order experiments...")
        plot_std_vs_time_qwak(
            tesselation_stds, 
            shift_probs, 
            title_prefix="Tesselation shift (mean)", 
            parameter_name="prob"
        )
    else:
        print("\nNo tesselation order data to plot.")

def main():
    """
    Main function to load experiments and create plots.
    """
    print("="*60)
    print("STANDARD DEVIATION COMPARISON")
    print("="*60)
    
    base_dir = "experiments_data_samples_probDist"
    
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
    
    # Create plots
    print("\n" + "-"*40)
    print("CREATING PLOTS")
    print("-"*40)
    
    # Combined comparison plot
    print("\nCreating combined comparison plot...")
    plot_combined_comparison(angle_stds, angle_devs, tesselation_stds, shift_probs)
    
    # Individual plots using original plotting function
    plot_individual_experiments(angle_stds, angle_devs, tesselation_stds, shift_probs)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
