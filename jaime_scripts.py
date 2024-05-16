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
