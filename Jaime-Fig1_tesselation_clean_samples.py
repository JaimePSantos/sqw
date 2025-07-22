from sqw.tesselations import even_cycle_two_tesselation,even_line_two_tesselation
from sqw.experiments_expanded import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

from utils.plotTools import plot_qwak
from jaime_scripts import (
    get_experiment_dir, 
    run_and_save_experiment_generic, 
    load_experiment_results_generic,
    load_or_create_experiment_generic,
    plot_multiple_timesteps_qwak,
    plot_std_vs_time_qwak,
    plot_single_timestep_qwak,
    # Import the improved sample-based functions
    run_and_save_experiment_samples,
    load_experiment_results_samples,
    load_or_create_experiment_samples,
    create_mean_probability_distributions,
    load_mean_probability_distributions,
    check_mean_probability_distributions_exist,
    load_or_create_mean_probability_distributions,
    prob_distributions2std,
    smart_load_or_create_experiment  # New intelligent loading function
)

import networkx as nx
import numpy as np
import os
import pickle

# All experiment functions are now imported from jaime_scripts to avoid duplication

# Convenience wrapper for tesselation-specific use case
def run_and_save_experiment(graph_func, tesselation_func, N, steps, tesselation_orders_list, angles, initial_state_func, initial_state_kwargs, shift_probs, base_dir="experiments_data"):
    """Wrapper for run_and_save_experiment_generic with tesselation-specific settings."""
    noise_params_list = [[prob, prob] if prob > 0 else [0, 0] for prob in shift_probs]
    return run_and_save_experiment_generic(
        graph_func=graph_func, tesselation_func=tesselation_func, N=N, steps=steps,
        parameter_list=shift_probs, angles_or_angles_list=angles, tesselation_order_or_list=tesselation_orders_list,
        initial_state_func=initial_state_func, initial_state_kwargs=initial_state_kwargs,
        noise_params_list=noise_params_list, noise_type="tesselation_order", parameter_name="prob", base_dir=base_dir
    )

def load_experiment_results(tesselation_func, N, steps, shift_probs, base_dir="experiments_data"):
    """Wrapper for load_experiment_results_generic with tesselation-specific settings."""
    noise_params_list = [[prob, prob] if prob > 0 else [0, 0] for prob in shift_probs]
    return load_experiment_results_generic(tesselation_func, N, steps, shift_probs, noise_params_list, "tesselation_order", base_dir)

def load_or_create_experiment(graph_func, tesselation_func, N, steps, tesselation_orders_list, angles, initial_state_func, initial_state_kwargs, shift_probs, base_dir="experiments_data"):
    """Wrapper for smart_load_or_create_experiment with tesselation-specific settings."""
    return smart_load_or_create_experiment(
        graph_func=graph_func, tesselation_func=tesselation_func, N=N, steps=steps,
        angles_or_angles_list=angles,  # Fixed angles for all experiments
        tesselation_order_or_list=tesselation_orders_list,  # 2D list [shift_prob][sample] -> tesselation_order
        initial_state_func=initial_state_func, initial_state_kwargs=initial_state_kwargs,
        parameter_list=shift_probs, samples=len(tesselation_orders_list[0]) if tesselation_orders_list else None, 
        noise_type="tesselation_order", parameter_name="prob",
        samples_base_dir=base_dir + "_samples", probdist_base_dir=base_dir + "_samples_probDist"
    )

if __name__ == "__main__":
    N = 100
    steps = N//4
    samples = 10  # Number of samples per shift probability
    angles = [[np.pi/3, np.pi/3]] * steps  # Fixed angles, no noise
    initial_state_kwargs = {"nodes": [N//2]}

    # List of shift probabilities for tesselation order switching
    shift_probs = [0, 0.2, 0.5]
    
    # Generate tesselation orders for each shift probability and sample
    tesselation_orders_list = []  # [shift_prob][sample] -> tesselation_order
    for shift_prob in shift_probs:
        shift_tesselation_orders = []
        for sample_idx in range(samples):
            if shift_prob == 0:
                # No noise case - use fixed tesselation order
                shift_tesselation_orders.append([[0, 1]] * steps)
            else:
                # Noisy tesselation order - generate different random sequence for each sample
                tesselation_order = tesselation_choice([[0, 1], [1, 0]], steps, [1 - shift_prob, shift_prob])
                shift_tesselation_orders.append(tesselation_order)
        tesselation_orders_list.append(shift_tesselation_orders)

    print(f"Running experiment for {len(shift_probs)} different tesselation shift probabilities with {samples} samples each...")
    print(f"Shift probabilities: {shift_probs}")

    # Use the new smart loading function instead of the old approach
    print("Using smart loading (probabilities → samples → create)...")
    results_list = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        tesselation_orders_list=tesselation_orders_list,
        angles=angles,  # Fixed angles for all experiments
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        shift_probs=shift_probs
    )

    print(f"Got results for {len(results_list)} tesselation experiments")

    # Calculate statistics for plotting using mean results (probability distributions)
    domain = np.arange(N)
    stds = []
    for i, mean_prob_dists in enumerate(results_list):
        if mean_prob_dists and len(mean_prob_dists) > 0 and all(state is not None for state in mean_prob_dists):
            # These should be probability distributions from smart loading
            std_values = prob_distributions2std(mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Tesselation {i} (shift_prob={shift_probs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Tesselation {i} (shift_prob={shift_probs[i]:.3f}): No valid mean probability distributions")
            stds.append([])

    # Plot all tesselation shifts in a single figure
    plot_std_vs_time_qwak(stds, shift_probs, title_prefix="Tesselation shift (mean)", parameter_name="prob")

    # # Plot probability distributions at specific timesteps
    # timestep_to_plot = steps // 2  # Middle timestep
    # print(f"\nPlotting distributions at timestep {timestep_to_plot}")
    # plot_single_timestep_qwak(results_list, shift_probs, timestep_to_plot, domain, "Tesselation shift", "prob")

    # # Plot distributions at multiple timesteps
    # timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    # print(f"\nPlotting distributions at timesteps {timesteps_to_plot}")
    # plot_multiple_timesteps_qwak(results_list, shift_probs, timesteps_to_plot, domain, "Tesselation shift", "prob")
