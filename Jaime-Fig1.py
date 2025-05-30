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
    """
    tesselation_name = tesselation_func.__name__
    noise_str = "noise" if has_noise else "nonoise"
    folder = f"{tesselation_name}_{noise_str}"
    subfolder = f"N_{N}"
    if has_noise and noise_params is not None:
        noise_suffix = "_".join(str(x) for x in noise_params)
        folder = f"{folder}_dev_{noise_suffix}"
    return os.path.join(base_dir, folder, subfolder)

def run_and_save_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles,
    tesselation_order,
    initial_state_func,
    initial_state_kwargs,
    angles_noisy=None,
    noise_params=None,
    base_dir="experiments_data"
):
    """
    Runs the experiment and saves each final state as a separate file in a structured folder.
    The parent folder encodes the total dev value, while the filename encodes the actual angles_noisy values.
    """
    # Save no-noise results
    exp_dir = get_experiment_dir(tesselation_func, False, N, base_dir=base_dir)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[run_and_save_experiment] Saving no-noise results to {exp_dir}")

    G = graph_func(N)
    T = tesselation_func(N)
    initial_state = initial_state_func(N, **initial_state_kwargs)

    print("[run_and_save_experiment] Running without noise...")
    final_states_no_noise = running(
        G, T, steps,
        initial_state,
        angles=angles,
        tesselation_order=tesselation_order
    )
    for i, state in enumerate(final_states_no_noise):
        filename = f"final_state_step_{i}.pkl"
        with open(os.path.join(exp_dir, filename), "wb") as f:
            pickle.dump(state, f)
    print(f"[run_and_save_experiment] Saved {len(final_states_no_noise)} no-noise states.")

    # Save noise results if provided
    final_states_noise = None
    if angles_noisy is not None:
        exp_dir_noise = get_experiment_dir(tesselation_func, True, N, noise_params=noise_params, base_dir=base_dir)
        os.makedirs(exp_dir_noise, exist_ok=True)
        print(f"[run_and_save_experiment] Saving noise results to {exp_dir_noise}")
        print("[run_and_save_experiment] Running with noise...")
        final_states_noise = running(
            G, T, steps,
            initial_state,
            angles=angles_noisy,
            tesselation_order=tesselation_order
        )
        num_states = len(final_states_noise)
        num_angles = len(angles_noisy)
        for i, state in enumerate(final_states_noise):
            if i < num_angles:
                angle_str = "_".join(f"{v:.6f}" for v in np.ravel(angles_noisy[i]))
            else:
                # Use the last available angles if out of range
                angle_str = "_".join(f"{v:.6f}" for v in np.ravel(angles_noisy[-1]))
            filename = f"final_state_step_{i}_angles_{angle_str}.pkl"
            with open(os.path.join(exp_dir_noise, filename), "wb") as f:
                pickle.dump(state, f)
        print(f"[run_and_save_experiment] Saved {len(final_states_noise)} noise states.")

    return {
        "final_states_no_noise": final_states_no_noise,
        "final_states_noise": final_states_noise
    }

def load_experiment_results(
    tesselation_func,
    N,
    steps,
    has_noise=False,
    noise_params=None,
    angles_noisy=None,
    base_dir="experiments_data"
):
    """
    Loads all final states from disk for the given experiment setup.
    For noisy case, expects angles_noisy to reconstruct the filenames.
    """
    exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, base_dir=base_dir)
    results = []
    for i in range(steps):
        if has_noise and angles_noisy is not None:
            angle_str = "_".join(f"{v:.6f}" for v in np.ravel(angles_noisy[i]))
            filename = f"final_state_step_{i}_angles_{angle_str}.pkl"
        else:
            filename = f"final_state_step_{i}.pkl"
        filepath = os.path.join(exp_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                results.append(pickle.load(f))
        else:
            print(f"[load_experiment_results] File {filepath} does not exist.")
            results.append(None)
    return results

def load_or_create_experiment(
    graph_func,
    tesselation_func,
    N,
    steps,
    angles,
    tesselation_order,
    initial_state_func,
    initial_state_kwargs,
    angles_noisy=None,
    noise_params=None,
    base_dir="experiments_data"
):
    """
    Loads experiment results if they exist, otherwise runs and saves the experiment.
    Returns a dictionary with the results.
    """
    # Check for no-noise
    exp_dir_no_noise = get_experiment_dir(tesselation_func, False, N, base_dir=base_dir)
    no_noise_exists = all(
        os.path.exists(os.path.join(exp_dir_no_noise, f"final_state_step_{i}.pkl"))
        for i in range(steps)
    )
    # Check for noise
    if angles_noisy is not None:
        exp_dir_noise = get_experiment_dir(tesselation_func, True, N, noise_params=noise_params, base_dir=base_dir)
        noise_exists = all(
            os.path.exists(os.path.join(
                exp_dir_noise,
                f"final_state_step_{i}_angles_{'_'.join(f'{v:.6f}' for v in np.ravel(angles_noisy[i]))}.pkl"
            ))
            for i in range(steps)
        )
    else:
        noise_exists = True

    if no_noise_exists and noise_exists:
        print("[load_or_create_experiment] All files found, loading results.")
        final_states_no_noise = load_experiment_results(
            tesselation_func, N, steps, has_noise=False, base_dir=base_dir
        )
        final_states_noise = None
        if angles_noisy is not None:
            final_states_noise = load_experiment_results(
                tesselation_func, N, steps, has_noise=True, noise_params=noise_params, angles_noisy=angles_noisy, base_dir=base_dir
            )
        return {
            "final_states_no_noise": final_states_no_noise,
            "final_states_noise": final_states_noise
        }
    else:
        print("[load_or_create_experiment] Some files missing, running experiment and saving results.")
        return run_and_save_experiment(
            graph_func=graph_func,
            tesselation_func=tesselation_func,
            N=N,
            steps=steps,
            angles=angles,
            tesselation_order=tesselation_order,
            initial_state_func=initial_state_func,
            initial_state_kwargs=initial_state_kwargs,
            angles_noisy=angles_noisy,
            noise_params=noise_params,
            base_dir=base_dir
        )

# Example usage:
if __name__ == "__main__":
    N = 100
    steps = N
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}
    noise_params = [1000, 1000]
    angles_noisy = random_angle_deviation([np.pi/3, np.pi/3], noise_params, steps)
    experiment_name = "cycle_vs_line"

    results = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles=angles,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        angles_noisy=angles_noisy,
        noise_params=noise_params
    )

    # Calculate statistics for plotting
    domain = np.arange(N)
    std_no_noise = states2std(results["final_states_no_noise"], domain)
    std_noise = states2std(results["final_states_noise"], domain)
    initial_node = N // 2
    survival_no_noise = states2survival(results["final_states_no_noise"], initial_node)
    survival_noise = states2survival(results["final_states_noise"], initial_node)

    # # Plot standard deviation vs time
    # plt.figure(figsize=(8, 5))
    # plt.plot(std_no_noise, label='No Noise')
    # plt.plot(std_noise, label='With Noise')
    # plt.xlabel('Time Step')
    # plt.ylabel('Standard Deviation')
    # plt.title(f'Standard Deviation vs Time after {steps} Steps (Noisy)' )
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Plot survival probability vs time (log-log scale)
    # plt.figure(figsize=(8, 5))
    # plt.loglog(survival_no_noise, label='No Noise')
    # plt.loglog(survival_noise, label='With Noise')
    # plt.xlabel('Time Step')
    # plt.ylabel('Survival Probability')
    # plt.title(f'Survival Probability vs Time (Log-Log) after {steps} Steps (Noisy)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # Plot probability distribution after 80 steps for the noisy case
    # prob_dist_noise = np.abs(results["final_states_noise"][-1])**2
    # plt.figure(figsize=(8, 5))
    # plt.plot(domain, prob_dist_noise, label='Noisy Case')
    # plt.xlabel('Node')
    # plt.ylabel('Probability')
    # plt.title(f'Probability Distribution after {steps} Steps (Noisy)')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # --- Extra plot: Compare no noise, dev=0.698..., dev=pi/3, dev=100 ---
    # Load dev=0.698...
    dev_val_0698 = (np.pi/3)/1.5
    noise_params_0698 = [dev_val_0698, dev_val_0698]
    angles_noisy_0698 = random_angle_deviation([np.pi/3, np.pi/3], noise_params_0698, steps)
    results_dev_0698 = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles=angles,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        angles_noisy=angles_noisy_0698,
        noise_params=noise_params_0698
    )
    std_dev_0698 = states2std(results_dev_0698["final_states_noise"], domain)

    # Load dev=pi/3
    dev_val_pi3 = np.pi/3
    noise_params_pi3 = [dev_val_pi3, dev_val_pi3]
    angles_noisy_pi3 = random_angle_deviation([np.pi/3, np.pi/3], noise_params_pi3, steps)
    results_dev_pi3 = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles=angles,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        angles_noisy=angles_noisy_pi3,
        noise_params=noise_params_pi3
    )
    std_dev_pi3 = states2std(results_dev_pi3["final_states_noise"], domain)

    # Load dev=100
    noise_params_100 = [100, 100]
    angles_noisy_100 = random_angle_deviation([np.pi/3, np.pi/3], noise_params_100, steps)
    results_dev_100 = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles=angles,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        angles_noisy=angles_noisy_100,
        noise_params=noise_params_100
    )
    std_dev_100 = states2std(results_dev_100["final_states_noise"], domain)

    # Load dev=1000
    noise_params_1000 = [1000, 1000]
    angles_noisy_1000 = random_angle_deviation([np.pi/3, np.pi/3], noise_params_1000, steps)
    results_dev_1000 = load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles=angles,
        tesselation_order=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        angles_noisy=angles_noisy_1000,
        noise_params=noise_params_1000
    )
    std_dev_1000 = states2std(results_dev_1000["final_states_noise"], domain)

    # Plot all five
    plt.figure(figsize=(8, 5))
    plt.plot(std_no_noise, label='No Noise')
    plt.plot(std_dev_0698, label='Noise dev=0.698...')
    plt.plot(std_dev_pi3, label='Noise dev=pi/3')
    plt.plot(std_dev_100, label='Noise dev=100')
    plt.plot(std_dev_1000, label='Noise dev=1000')
    plt.xlabel('Time Step')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation vs Time for Different Noise Deviations')
    plt.legend()
    plt.tight_layout()
    plt.show()
