from sqw.tesselations import even_cycle_two_tesselation,even_line_two_tesselation
from sqw.experiments_expanded import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    Plot probability distributions for all noise levels at a specific timestep.
    """
    plt.figure(figsize=(10, 6))
    for i, (walk_states, dev) in enumerate(zip(results_list, devs)):
        if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
            state = walk_states[timestep]
            prob_dist = np.abs(state)**2
            plt.plot(domain, prob_dist, label=f'{title_prefix} noise dev={dev:.3f}', alpha=0.8)
        else:
            print(f"No valid state for dev={dev:.3f} at timestep {timestep}")
    
    plt.xlabel('Position')
    plt.ylabel('Probability')
    plt.title(f'Probability Distributions at Timestep {timestep}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multiple_timesteps(results_list, devs, timesteps, domain, title_prefix="Angle"):
    """
    Plot probability distributions for multiple timesteps in subplots.
    """
    n_timesteps = len(timesteps)
    cols = min(3, n_timesteps)
    rows = (n_timesteps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_timesteps == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, timestep in enumerate(timesteps):
        ax = axes[idx]
        for i, (walk_states, dev) in enumerate(zip(results_list, devs)):
            if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
                state = walk_states[timestep]
                prob_dist = np.abs(state)**2
                ax.plot(domain, prob_dist, label=f'dev={dev:.3f}', alpha=0.8)
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability')
        ax.set_title(f'Timestep {timestep}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_timesteps, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'{title_prefix} Noise: Probability Distributions at Different Timesteps')
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    N = 100
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
    plt.figure(figsize=(8, 5))
    for std, dev in zip(stds, devs):
        if len(std) > 0:
            plt.plot(std, label=f'Angle noise dev={dev:.3f}')
    plt.xlabel('Time Step')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Time for Different Angle Noise Deviations')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot probability distributions at specific timesteps
    timestep_to_plot = steps // 2  # Middle timestep
    print(f"\nPlotting distributions at timestep {timestep_to_plot}")
    plot_distributions_at_timestep(results_list, devs, timestep_to_plot, domain, "Angle")
    
    # Plot distributions at multiple timesteps
    timesteps_to_plot = [0, steps//4, steps//2, 3*steps//4, steps-1]
    print(f"\nPlotting distributions at timesteps {timesteps_to_plot}")
    plot_multiple_timesteps(results_list, devs, timesteps_to_plot, domain, "Angle")
