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

# All experiment functions are now imported from jaime_scripts to avoid duplication

# Convenience wrapper for this specific use case
def run_and_save_experiment(graph_func, tesselation_func, N, steps, angles_list, tesselation_order, initial_state_func, initial_state_kwargs, devs, base_dir="experiments_data"):
    """Wrapper for run_and_save_experiment_generic with angle-specific settings."""
    noise_params_list = [[dev, dev] if dev > 0 else [0, 0] for dev in devs]
    return run_and_save_experiment_generic(
        graph_func=graph_func, tesselation_func=tesselation_func, N=N, steps=steps, 
        parameter_list=devs, angles_or_angles_list=angles_list, tesselation_order_or_list=tesselation_order,
        initial_state_func=initial_state_func, initial_state_kwargs=initial_state_kwargs,
        noise_params_list=noise_params_list, noise_type="angle", parameter_name="dev", base_dir=base_dir
    )

def load_experiment_results(tesselation_func, N, steps, devs, base_dir="experiments_data"):
    """Wrapper for load_experiment_results_generic with angle-specific settings."""
    noise_params_list = [[dev, dev] if dev > 0 else [0, 0] for dev in devs]
    return load_experiment_results_generic(tesselation_func, N, steps, devs, noise_params_list, "angle", base_dir)

def load_or_create_experiment(graph_func, tesselation_func, N, steps, angles_list, tesselation_order, initial_state_func, initial_state_kwargs, devs, base_dir="experiments_data"):
    """Wrapper for load_or_create_experiment_generic with angle-specific settings."""
    noise_params_list = [[dev, dev] if dev > 0 else [0, 0] for dev in devs]
    return load_or_create_experiment_generic(
        graph_func=graph_func, tesselation_func=tesselation_func, N=N, steps=steps,
        parameter_list=devs, angles_or_angles_list=angles_list, tesselation_order_or_list=tesselation_order,
        initial_state_func=initial_state_func, initial_state_kwargs=initial_state_kwargs,
        noise_params_list=noise_params_list, noise_type="angle", parameter_name="dev", base_dir=base_dir
    )

if __name__ == "__main__":
    N = 100
    steps = N//4
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # List of devs and corresponding angles
    devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5]
    angles_list = []
    for dev in devs:
        if dev == 0:
            # No noise case - use perfect angles
            angles_list.append([[np.pi/3, np.pi/3]] * steps)
        else:
            angles_list.append(random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps))

    print(f"Running experiment for {len(devs)} different angle noise deviations...")
    print(f"Angle devs: {devs}")

    results_list = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_list=angles_list,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        devs=devs
    )

    print(f"Got results for {len(results_list)} walks")

    # Calculate statistics for plotting
    domain = np.arange(N)
    stds = []
    for i, walk_states in enumerate(results_list):
        if walk_states and len(walk_states) > 0 and all(state is not None for state in walk_states):
            std_values = states2std(walk_states, domain)
            stds.append(std_values)
            print(f"Walk {i} (angle_dev={devs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Walk {i} (angle_dev={devs[i]:.3f}): No valid states")
            stds.append([])

    # Plot all walks in a single figure
    plot_std_vs_time_qwak(stds, devs, title_prefix="Angle noise", parameter_name="dev")

    # Plot probability distributions at specific timesteps
    timestep_to_plot = steps // 2  # Middle timestep
    print(f"\nPlotting distributions at timestep {timestep_to_plot}")
    plot_single_timestep_qwak(results_list, devs, timestep_to_plot, domain, "Angle noise", "dev")

    # Plot distributions at multiple timesteps
    timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    print(f"\nPlotting distributions at timesteps {timesteps_to_plot}")
    plot_multiple_timesteps_qwak(results_list, devs, timesteps_to_plot, domain, "Angle noise", "dev")
