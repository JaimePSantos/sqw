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

if __name__ == "__main__":
    N = 2000
    steps = N//4
    samples = 10  # Number of samples per deviation
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    # List of devs and corresponding angles_list_list
    # devs = [0, (np.pi/3)/2.5, (np.pi/3)/2, (np.pi/3), (np.pi/3) * 1.5,(np.pi/3) * 2]
    devs = [0, (np.pi/3)/2.5,(np.pi/3) * 2]
    angles_list_list = []  # [dev][sample] -> angles
    
    for dev in devs:
        dev_angles_list = []
        for sample_idx in range(samples):
            if dev == 0:
                # No noise case - use perfect angles
                dev_angles_list.append([[np.pi/3, np.pi/3]] * steps)
            else:
                dev_angles_list.append(random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps))
        angles_list_list.append(dev_angles_list)

    print(f"Running experiment for {len(devs)} different angle noise deviations with {samples} samples each...")
    print(f"Angle devs: {devs}")

    # Use the new smart loading function instead of the old approach
    print("Using smart loading (probabilities → samples → create)...")
    mean_results = smart_load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_or_angles_list=angles_list_list,
        tesselation_order_or_list=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=devs,
        samples=samples,
        noise_type="angle",
        parameter_name="angle_dev",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )

    # Calculate statistics for plotting using mean results
    domain = np.arange(N)
    stds = []
    for i, dev_mean_prob_dists in enumerate(mean_results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0 and all(state is not None for state in dev_mean_prob_dists):
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Dev {i} (angle_dev={devs[i]:.2f}): {len(std_values)} std values")
        else:
            print(f"Dev {i} (angle_dev={devs[i]:.2f}): No valid mean probability distributions")
            stds.append([])

    # Plot all devs in a single figure
    plot_std_vs_time_qwak(stds, devs, title_prefix="Angle noise (mean)", parameter_name="dev")

    # # Plot probability distributions at specific timesteps
    # timestep_to_plot = steps // 2  # Middle timestep
    # print(f"\nPlotting mean distributions at timestep {timestep_to_plot}")
    # plot_single_timestep_qwak(mean_results, devs, timestep_to_plot, domain, "Angle noise (mean)", "dev")

    # # Plot distributions at multiple timesteps
    # timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    # print(f"\nPlotting mean distributions at timesteps {timesteps_to_plot}")
    # plot_multiple_timesteps_qwak(mean_results, devs, timesteps_to_plot, domain, "Angle noise (mean)", "dev")
