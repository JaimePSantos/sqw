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
    plot_single_timestep_qwak
)

import networkx as nx
import numpy as np
import os
import pickle

def run_and_save_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    tesselation_orders_list,  # List of tesselation orders for each walk
    angles,  # Fixed angles (no noise)
    initial_state_func,
    initial_state_kwargs,
    shift_probs,  # List of shift probabilities for each walk
    base_dir="experiments_data"
):
    """
    Runs the experiment for each tesselation_order/shift_prob and saves each walk's final states in its own shift_prob folder.
    """
    noise_params_list = [[shift_prob] if shift_prob > 0 else [0] for shift_prob in shift_probs]
    return run_and_save_experiment_generic(
        graph_func=graph_func,
        tesselation_func=tesselation_func,
        N=N,
        steps=steps,
        parameter_list=shift_probs,
        angles_or_angles_list=angles,
        tesselation_order_or_list=tesselation_orders_list,
        initial_state_func=initial_state_func,
        initial_state_kwargs=initial_state_kwargs,
        noise_params_list=noise_params_list,
        noise_type="tesselation_order",
        parameter_name="shift_prob",
        base_dir=base_dir
    )

def load_experiment_results(
    tesselation_func,
    N,
    steps,
    shift_probs,
    tesselation_orders_list,
    base_dir="experiments_data"
):
    """
    Loads all final states from disk for each shift_prob/tesselation_order in the list.
    """
    noise_params_list = [[shift_prob] if shift_prob > 0 else [0] for shift_prob in shift_probs]
    return load_experiment_results_generic(
        tesselation_func=tesselation_func,
        N=N,
        steps=steps,
        parameter_list=shift_probs,
        noise_params_list=noise_params_list,
        noise_type="tesselation_order",
        base_dir=base_dir
    )

def load_or_create_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    tesselation_orders_list,  # List of tesselation orders for each walk
    angles,  # Fixed angles (no noise)
    initial_state_func,
    initial_state_kwargs,
    shift_probs,  # List of shift probabilities for each walk
    base_dir="experiments_data"
):
    """
    Loads experiment results for each walk if they exist, otherwise runs and saves them.
    Returns a list of lists: [walk1_states, walk2_states, ...]
    """
    noise_params_list = [[shift_prob] if shift_prob > 0 else [0] for shift_prob in shift_probs]
    return load_or_create_experiment_generic(
        graph_func=graph_func,
        tesselation_func=tesselation_func,
        N=N,
        steps=steps,
        parameter_list=shift_probs,
        angles_or_angles_list=angles,
        tesselation_order_or_list=tesselation_orders_list,
        initial_state_func=initial_state_func,
        initial_state_kwargs=initial_state_kwargs,
        noise_params_list=noise_params_list,
        noise_type="tesselation_order",
        parameter_name="shift_prob",
        base_dir=base_dir
    )



# Example usage:
if __name__ == "__main__":
    N = 1000
    steps = N//4
    angles = [[np.pi/3, np.pi/3]] * steps  # Fixed angles, no noise
    initial_state_kwargs = {"nodes": [N//2]}

    # List of shift probabilities for tesselation order switching
    shift_probs = [0, 0.1, 0.2, 0.3, 0.5,0.8]
    
    # Generate tesselation orders for each shift probability
    tesselation_orders_list = []
    for shift_prob in shift_probs:
        if shift_prob == 0:
            # No noise case - use fixed tesselation order
            tesselation_orders_list.append([[0, 1]] * steps)
        else:
            # Noisy tesselation order
            tesselation_order = tesselation_choice([[0, 1], [1, 0]], steps, [1 - shift_prob, shift_prob])
            tesselation_orders_list.append(tesselation_order)

    print(f"Running experiment for {len(shift_probs)} different tesselation shift probabilities...")
    print(f"Shift probabilities: {shift_probs}")

    results_list = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        tesselation_orders_list=tesselation_orders_list,
        angles=angles,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        shift_probs=shift_probs
    )

    print(f"Got results for {len(results_list)} walks")

    # Calculate statistics for plotting
    domain = np.arange(N)
    stds = []
    for i, walk_states in enumerate(results_list):
        if walk_states and len(walk_states) > 0 and all(state is not None for state in walk_states):
            std_values = states2std(walk_states, domain)
            stds.append(std_values)
            print(f"Walk {i} (shift_prob={shift_probs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Walk {i} (shift_prob={shift_probs[i]:.3f}): No valid states")
            stds.append([])

    # Plot all walks in a single figure
    plot_std_vs_time_qwak(stds, shift_probs, title_prefix="Tesselation shift", parameter_name="prob")

    # Plot probability distributions at specific timesteps
    timestep_to_plot = steps // 2  # Middle timestep
    print(f"\nPlotting distributions at timestep {timestep_to_plot}")
    plot_single_timestep_qwak(results_list, shift_probs, timestep_to_plot, domain, "Tesselation shift", "prob")

    # Plot distributions at multiple timesteps
    timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    print(f"\nPlotting distributions at timesteps {timesteps_to_plot}")
    plot_multiple_timesteps_qwak(results_list, shift_probs, timesteps_to_plot, domain, "Tesselation shift", "prob")
