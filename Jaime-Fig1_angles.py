from sqw.tesselations import even_cycle_two_tesselation,even_line_two_tesselation
from sqw.experiments_expanded import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

from utils.plotTools import plot_qwak

import networkx as nx
import numpy as np
import os
import pickle

def get_experiment_dir(
    tesselation_func,
    has_noise,
    N,
    noise_params=None,
    base_dir="experiments_data"
):
    """
    Returns the directory path for the experiment based on tesselation, noise, and graph size.
    Now, dev is a subfolder inside the tesselation_angle_noise/N_{N} folder.
    """
    tesselation_name = tesselation_func.__name__
    noise_str = "angle_noise" if has_noise else "angle_nonoise"
    folder = f"{tesselation_name}_{noise_str}"
    base = os.path.join(base_dir, folder, f"N_{N}")
    if has_noise and noise_params is not None:
        # Round each noise param to 2 decimal places for folder name
        noise_suffix = "_".join(f"{float(x):.2f}" for x in noise_params)
        dev_folder = f"angle_dev_{noise_suffix}"
        return os.path.join(base, dev_folder)
    else:
        return base

def run_and_save_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles_list,  # List of angles for each walk
    tesselation_order,
    initial_state_func,
    initial_state_kwargs,
    devs,  # List of devs for each walk
    base_dir="experiments_data"
):
    """
    Runs the experiment for each angles_list/dev and saves each walk's final states in its own dev folder.
    """
    results = []
    for angles_noisy, dev in zip(angles_list, devs):
        has_noise = dev > 0
        noise_params = [dev, dev] if has_noise else None
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, base_dir=base_dir)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[run_and_save_experiment] Saving results to {exp_dir} for angle_dev={dev:.3f}")

        G = graph_func(N)
        T = tesselation_func(N)
        initial_state = initial_state_func(N, **initial_state_kwargs)

        print("[run_and_save_experiment] Running walk...")
        final_states = running(
            G, T, steps,
            initial_state,
            angles=angles_noisy,
            tesselation_order=tesselation_order
        )
        for i, state in enumerate(final_states):
            filename = f"final_state_step_{i}.pkl"
            with open(os.path.join(exp_dir, filename), "wb") as f:
                pickle.dump(state, f)
        print(f"[run_and_save_experiment] Saved {len(final_states)} states for angle_dev={dev:.3f}.")
        results.append(final_states)
    return results

def load_experiment_results(
    tesselation_func,
    N,
    steps,
    devs,
    angles_list,
    base_dir="experiments_data"
):
    """
    Loads all final states from disk for each dev/angles in the list.
    """
    all_results = []
    for angles_noisy, dev in zip(angles_list, devs):
        has_noise = dev > 0
        noise_params = [dev, dev] if has_noise else None
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, base_dir=base_dir)
        walk_results = []
        for i in range(steps):
            filename = f"final_state_step_{i}.pkl"
            filepath = os.path.join(exp_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    walk_results.append(pickle.load(f))
            else:
                print(f"[load_experiment_results] File {filepath} does not exist.")
                walk_results.append(None)
        all_results.append(walk_results)
    return all_results

def load_or_create_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles_list,  # List of angles for each walk
    tesselation_order,
    initial_state_func,
    initial_state_kwargs,
    devs,  # List of devs for each walk
    base_dir="experiments_data"
):
    """
    Loads experiment results for each walk if they exist, otherwise runs and saves them.
    Returns a list of lists: [walk1_states, walk2_states, ...]
    """
    # Check for each walk
    all_exists = []
    for angles_noisy, dev in zip(angles_list, devs):
        has_noise = dev > 0
        noise_params = [dev, dev] if has_noise else None
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, base_dir=base_dir)
        exists = all(
            os.path.exists(os.path.join(exp_dir, f"final_state_step_{i}.pkl"))
            for i in range(steps)
        )
        all_exists.append(exists)

    if all(all_exists):
        print("[load_or_create_experiment] All files found, loading results.")
        return load_experiment_results(
            tesselation_func, N, steps, devs, angles_list, base_dir=base_dir
        )
    else:
        print("[load_or_create_experiment] Some files missing, running experiment and saving results.")
        return run_and_save_experiment(
            graph_func=graph_func,
            tesselation_func=tesselation_func,
            N=N,
            steps=steps,
            angles_list=angles_list,
            tesselation_order=tesselation_order,
            initial_state_func=initial_state_func,
            initial_state_kwargs=initial_state_kwargs,
            devs=devs,
            base_dir=base_dir
        )


def plot_distributions_at_timestep(results_list, devs, timestep, domain, title_prefix="Angle"):
    """
    Plot probability distributions for all noise levels at a specific timestep using plot_qwak.
    """
    y_value_matrix = []
    legend_labels = []
    for walk_states, dev in zip(results_list, devs):
        if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
            state = walk_states[timestep]
            prob_dist = np.abs(state)**2
            y_value_matrix.append(prob_dist)
            legend_labels.append(f'{title_prefix} noise dev={dev:.3f}')
        else:
            print(f"No valid state for dev={dev:.3f} at timestep {timestep}")
    if y_value_matrix:
        x_value_matrix = [domain] * len(y_value_matrix)
        plot_qwak(
            x_value_matrix,
            y_value_matrix,
            x_label='Position',
            y_label='Probability',
            plot_title=f'Probability Distributions at Timestep {timestep}',
            legend_labels=legend_labels,
            legend_title=None,
            legend_ncol=1,
            legend_loc='best',
            use_grid=True,
            font_size=14,
            figsize=(10, 6)
        )

def plot_multiple_timesteps_qwak(results_list, devs, timesteps, domain, title_prefix="Angle"):
    """
    Plot probability distributions for multiple timesteps using plot_qwak.
    Each line is a walk, each color is a timestep, all in a single figure.
    """
    import matplotlib.pyplot as plt

    # For each walk, plot its distribution at each timestep as a separate line (color by timestep)
    for i, (walk_states, dev) in enumerate(zip(results_list, devs)):
        x_value_matrix = []
        y_value_matrix = []
        legend_labels = []
        for timestep in timesteps:
            if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
                state = walk_states[timestep]
                prob_dist = np.abs(state)**2
                x_value_matrix.append(domain)
                y_value_matrix.append(prob_dist)
                legend_labels.append(f't={timestep}')
        if y_value_matrix:
            plot_qwak(
                x_value_matrix,
                y_value_matrix,
                x_label='Position',
                y_label='Probability',
                plot_title=f'{title_prefix} Noise dev={dev:.3f}: Distributions at Multiple Timesteps',
                legend_labels=legend_labels,
                legend_title='Timestep',
                legend_ncol=1,
                legend_loc='best',
                use_grid=True,
                font_size=12,
                figsize=(8, 5)
            )

def plot_std_vs_time_qwak(stds, devs):
    """
    Plot standard deviation vs time for all walks using plot_qwak.
    """
    x_value_matrix = [list(range(len(std))) for std in stds if len(std) > 0]
    y_value_matrix = [std for std in stds if len(std) > 0]
    legend_labels = [f'Angle noise dev={dev:.3f}' for std, dev in zip(stds, devs) if len(std) > 0]
    if y_value_matrix:
        plot_qwak(
            x_value_matrix,
            y_value_matrix,
            x_label='Time Step',
            y_label='Standard Deviation',
            plot_title='Standard Deviation vs Time for Different Angle Noise Deviations',
            legend_labels=legend_labels,
            legend_title=None,
            legend_ncol=1,
            legend_loc='best',
            use_grid=True,
            font_size=14,
            figsize=(8, 5)
        )

def plot_single_timestep_qwak(results_list, devs, timestep, domain, title_prefix="Angle"):
    """
    Plot probability distributions for all noise levels at a specific timestep using plot_qwak.
    """
    x_value_matrix = []
    y_value_matrix = []
    legend_labels = []
    for walk_states, dev in zip(results_list, devs):
        if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
            state = walk_states[timestep]
            prob_dist = np.abs(state)**2
            x_value_matrix.append(domain)
            y_value_matrix.append(prob_dist)
            legend_labels.append(f'{title_prefix} noise dev={dev:.3f}')
    if y_value_matrix:
        plot_qwak(
            x_value_matrix,
            y_value_matrix,
            x_label='Position',
            y_label='Probability',
            plot_title=f'Probability Distributions at Timestep {timestep}',
            legend_labels=legend_labels,
            legend_title=None,
            legend_ncol=1,
            legend_loc='best',
            use_grid=True,
            font_size=14,
            figsize=(10, 6)
        )

# Example usage:
if __name__ == "__main__":
    N = 1000
    steps = N//3
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
    plot_std_vs_time_qwak(stds, devs)

    # Plot probability distributions at specific timesteps
    timestep_to_plot = steps // 2  # Middle timestep
    print(f"\nPlotting distributions at timestep {timestep_to_plot}")
    plot_single_timestep_qwak(results_list, devs, timestep_to_plot, domain, "Angle")

    # Plot distributions at multiple timesteps
    timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    print(f"\nPlotting distributions at timesteps {timesteps_to_plot}")
    plot_multiple_timesteps_qwak(results_list, devs, timesteps_to_plot, domain, "Angle")
