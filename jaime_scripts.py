import random
import networkx as nx
from matplotlib import pyplot as plt
from collections import Counter
import math
from utils.plotTools import plot_qwak
import os
import ast
import numpy as np
import json

def load_list_from_file(file_path):
    with open(file_path, 'r') as file:
        data_str = file.read()
    data = [json.loads(line) for line in data_str.splitlines()]
    return data


def write_list_to_file(file_path, data):
    data_str = [str(item) for item in data]  # Convert float values to strings
    with open(file_path, 'w') as file:
        file.write('\n'.join(data_str))
        
def load_or_generate_data(file1, file2, generation_func1, generation_func2, args1=(), kwargs1={}, args2=(), kwargs2={}):
    """
    Load data from files if they exist, or generate data using specified functions.
    
    :param file1: the file path to load the first data from
    :param file2: the file path to load the second data from
    :param generation_func1: the function to generate the first data if the file doesn't exist
    :param generation_func2: the function to generate the second data if the file doesn't exist
    :param args1: tuple containing positional arguments for the first generation function
    :param kwargs1: dict containing keyword arguments for the first generation function
    :param args2: tuple containing positional arguments for the second generation function
    :param kwargs2: dict containing keyword arguments for the second generation function
    :return: a tuple containing the two datasets
    """
    
    if os.path.exists(file1) and os.path.exists(file2):
        data1 = load_list_from_file(file1)
        data2 = load_list_from_file(file2)
        print('Files exist!')
    else:
        print('Files do not exist, generating data...')
        data1 = generation_func1(*args1, **kwargs1)
        data2 = generation_func2(*args2, **kwargs2)
        
        if not os.path.exists(file1):
            write_list_to_file(file1, data1)
            
        if not os.path.exists(file2):
            write_list_to_file(file2, data2)
    
    return data1, data2

def draw_graph(H, figsize=(8, 6), k=0.1, draw_self_loops=True, config={}):

    node_color = config.get('node_color', 'lightblue')
    node_size = config.get('node_size', 500)
    normal_edge_color = config.get('normal_edge_color', 'gray')
    normal_edge_width = config.get('normal_edge_width', 2.0)
    edge_style = config.get('edge_style', 'solid')
    self_loop_color = config.get('self_loop_color', 'red')
    title = config.get('title', '')
    draw_self_loops = config.get('draw_self_loops', True)
    figsize = config.get('figsize', (8, 6))
    
    plt.figure(figsize=figsize)
    
    pos = nx.spring_layout(H, k=k)
    # Get the weights of self-loops from adjacency matrix
    self_loop_weights = nx.to_numpy_array(H).diagonal()
    max_weight = max(self_loop_weights) if self_loop_weights.any() else 1

    # Set width relative to the max weight for self-loops
    self_loop_widths = {node: 4.0 * weight / max_weight for node, weight in zip(H.nodes, self_loop_weights)}

    # Prepare color and width for edges
    edge_colors = []
    edge_widths = []
    for u, v in H.edges():
        if u == v and draw_self_loops:  # self-loop
            edge_colors.append(self_loop_color)
            edge_widths.append(self_loop_widths[u])
        else:  # regular edge
            edge_colors.append(normal_edge_color)
            edge_widths.append(normal_edge_width)

    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_color=node_color, node_size=node_size)

    # Draw edges
    for edge, color, width in zip(H.edges(), edge_colors, edge_widths):
        if not (edge[0] == edge[1] and not draw_self_loops):  # Do not draw self-loops if draw_self_loops is False
            nx.draw_networkx_edges(H, pos, edgelist=[edge], edge_color=color, width=width, style=edge_style)

    # Draw labels
    nx.draw_networkx_labels(H, pos)

    plt.axis('off')
    plt.title(title)
    plt.show()
    
def draw_graph_from_adjacency_matrix(matrix):
    G = nx.from_numpy_array(matrix)
    nx.draw(G, with_labels=True)
    plt.show()
    
def print_matrix(matrix):
    # Print rows with left and right border
    for row in matrix:
        print('|', end='')
        print(' '.join(format(item, ".2f") for item in row), end=' |\n')


def save_array_list_to_file(array_list, filename):
    """
    Saves a list of arrays to a file.

    Parameters:
    - array_list: list of numpy arrays, the arrays to save.
    - filename: str, the name of the file to save the arrays.
    """
    with open(filename, 'w') as f:
        for array in array_list:
            # Save the shape of the array
            shape_str = ' '.join(map(str, array.shape))
            f.write(f'{shape_str}\n')
            # Save the array
            np.savetxt(f, array)
            f.write('\n')  # Separate arrays by an empty line


def load_array_list_from_file(filename):
    """
    Loads a list of arrays from a file.

    Parameters:
    - filename: str, the name of the file to load the arrays from.

    Returns:
    - list of numpy arrays, the loaded arrays.
    """
    array_list = []
    with open(filename, 'r') as f:
        content = f.read().strip().split('\n\n')  # Split by double newline
        for array_str in content:
            lines = array_str.splitlines()
            # First line contains the shape
            shape = tuple(map(int, lines[0].split()))
            # Remaining lines contain the array data
            array_data = '\n'.join(lines[1:])
            array = np.loadtxt(array_data.splitlines()).reshape(shape)
            array_list.append(array)
    return array_list

def get_experiment_dir(
    tesselation_func,
    has_noise,
    N,
    noise_params=None,
    noise_type="angle",  # "angle" or "tesselation_order"
    base_dir="experiments_data"
):
    """
    Returns the directory path for the experiment based on tesselation, noise, and graph size.
    """
    tesselation_name = tesselation_func.__name__
    if noise_type == "angle":
        noise_str = "angle_noise" if has_noise else "angle_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        if has_noise and noise_params is not None:
            # Round each noise param to 2 decimal places for folder name
            noise_suffix = "_".join(f"{float(x):.2f}" for x in noise_params)
            dev_folder = f"dev_{noise_suffix}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    elif noise_type == "tesselation_order":
        noise_str = "tesselation_order_noise" if has_noise else "tesselation_order_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        if has_noise and noise_params is not None:
            # Round each noise param to 3 decimal places for folder name
            noise_suffix = "_".join(f"{float(x):.3f}" for x in noise_params)
            shift_folder = f"tesselation_shift_prob_{noise_suffix}"
            return os.path.join(base, shift_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

def run_and_save_experiment_generic(
    graph_func,
    tesselation_func,
    N,
    steps,
    parameter_list,  # List of varying parameters for each walk
    angles_or_angles_list,  # Either fixed angles or list of angles for each walk
    tesselation_order_or_list,  # Either fixed tesselation_order or list for each walk
    initial_state_func,
    initial_state_kwargs,
    noise_params_list,  # List of noise parameters for each walk
    noise_type="angle",  # "angle" or "tesselation_order"
    parameter_name="dev",  # Name of the parameter for logging
    base_dir="experiments_data"
):
    """
    Generic function to run and save experiments for different parameter values.
    """
    from sqw.experiments_expanded import running
    import pickle
    
    results = []
    for i, (param, noise_params) in enumerate(zip(parameter_list, noise_params_list)):
        has_noise = any(p > 0 for p in noise_params) if isinstance(noise_params, list) else noise_params > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[run_and_save_experiment] Saving results to {exp_dir} for {parameter_name}={param:.3f}")

        G = graph_func(N)
        T = tesselation_func(N)
        initial_state = initial_state_func(N, **initial_state_kwargs)

        # Get the appropriate angles and tesselation_order for this walk
        if isinstance(angles_or_angles_list[0], list) and len(angles_or_angles_list) == len(parameter_list):
            angles = angles_or_angles_list[i]
        else:
            angles = angles_or_angles_list

        if isinstance(tesselation_order_or_list[0], list) and len(tesselation_order_or_list) == len(parameter_list):
            tesselation_order = tesselation_order_or_list[i]
        else:
            tesselation_order = tesselation_order_or_list

        print("[run_and_save_experiment] Running walk...")
        final_states = running(
            G, T, steps,
            initial_state,
            angles=angles,
            tesselation_order=tesselation_order
        )
        for j, state in enumerate(final_states):
            filename = f"final_state_step_{j}.pkl"
            with open(os.path.join(exp_dir, filename), "wb") as f:
                pickle.dump(state, f)
        print(f"[run_and_save_experiment] Saved {len(final_states)} states for {parameter_name}={param:.3f}.")
        results.append(final_states)
    return results

def load_experiment_results_generic(
    tesselation_func,
    N,
    steps,
    parameter_list,
    noise_params_list,
    noise_type="angle",
    base_dir="experiments_data"
):
    """
    Generic function to load experiment results from disk.
    """
    import pickle
    
    all_results = []
    for param, noise_params in zip(parameter_list, noise_params_list):
        has_noise = any(p > 0 for p in noise_params) if isinstance(noise_params, list) else noise_params > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
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

def load_or_create_experiment_generic(
    graph_func,
    tesselation_func,
    N,
    steps,
    parameter_list,
    angles_or_angles_list,
    tesselation_order_or_list,
    initial_state_func,
    initial_state_kwargs,
    noise_params_list,
    noise_type="angle",
    parameter_name="dev",
    base_dir="experiments_data"
):
    """
    Generic function to load experiment results if they exist, otherwise run and save them.
    """
    # Check for each walk
    all_exists = []
    for param, noise_params in zip(parameter_list, noise_params_list):
        has_noise = any(p > 0 for p in noise_params) if isinstance(noise_params, list) else noise_params > 0
        exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=base_dir)
        exists = all(
            os.path.exists(os.path.join(exp_dir, f"final_state_step_{i}.pkl"))
            for i in range(steps)
        )
        all_exists.append(exists)

    if all(all_exists):
        print("[load_or_create_experiment] All files found, loading results.")
        return load_experiment_results_generic(
            tesselation_func, N, steps, parameter_list, noise_params_list, noise_type=noise_type, base_dir=base_dir
        )
    else:
        print("[load_or_create_experiment] Some files missing, running experiment and saving results.")
        return run_and_save_experiment_generic(
            graph_func=graph_func,
            tesselation_func=tesselation_func,
            N=N,
            steps=steps,
            parameter_list=parameter_list,
            angles_or_angles_list=angles_or_angles_list,
            tesselation_order_or_list=tesselation_order_or_list,
            initial_state_func=initial_state_func,
            initial_state_kwargs=initial_state_kwargs,
            noise_params_list=noise_params_list,
            noise_type=noise_type,
            parameter_name=parameter_name,
            base_dir=base_dir
        )

def plot_multiple_timesteps_qwak(results_list, parameter_values, timesteps, domain, title_prefix="Parameter", parameter_name="param"):
    """
    Generic function to plot probability distributions for multiple timesteps using plot_qwak.
    Each line is a walk, each color is a timestep, all in a single figure.
    """
    import matplotlib.pyplot as plt

    # For each walk, plot its distribution at each timestep as a separate line (color by timestep)
    for i, (walk_states, param_val) in enumerate(zip(results_list, parameter_values)):
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
                plot_title=f'{title_prefix} {parameter_name}={param_val:.3f}: Distributions at Multiple Timesteps',
                legend_labels=legend_labels,
                legend_title='Timestep',
                legend_ncol=1,
                legend_loc='best',
                use_grid=True,
                font_size=12,
                figsize=(8, 5)
            )

def plot_std_vs_time_qwak(stds, parameter_values, title_prefix="Parameter", parameter_name="param"):
    """
    Generic function to plot standard deviation vs time for all walks using plot_qwak.
    """
    x_value_matrix = [list(range(len(std))) for std in stds if len(std) > 0]
    y_value_matrix = [std for std in stds if len(std) > 0]
    legend_labels = [f'{title_prefix} {parameter_name}={param_val:.3f}' for std, param_val in zip(stds, parameter_values) if len(std) > 0]
    if y_value_matrix:
        plot_qwak(
            x_value_matrix,
            y_value_matrix,
            x_label='Time Step',
            y_label='Standard Deviation',
            plot_title=f'Standard Deviation vs Time for Different {title_prefix} {parameter_name} Values',
            legend_labels=legend_labels,
            legend_title=None,
            legend_ncol=1,
            legend_loc='best',
            use_grid=True,
            font_size=14,
            figsize=(8, 5)
        )

def plot_single_timestep_qwak(results_list, parameter_values, timestep, domain, title_prefix="Parameter", parameter_name="param"):
    """
    Generic function to plot probability distributions for all parameter values at a specific timestep using plot_qwak.
    """
    x_value_matrix = []
    y_value_matrix = []
    legend_labels = []
    for walk_states, param_val in zip(results_list, parameter_values):
        if walk_states and len(walk_states) > timestep and walk_states[timestep] is not None:
            state = walk_states[timestep]
            prob_dist = np.abs(state)**2
            x_value_matrix.append(domain)
            y_value_matrix.append(prob_dist)
            legend_labels.append(f'{title_prefix} {parameter_name}={param_val:.3f}')
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
