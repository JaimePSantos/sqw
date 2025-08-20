from scipy.linalg import expm
from scipy import linalg
import numpy as np
from numpy import *
from itertools import combinations
import networkx as nx
import random

def uniform_initial_state(N, nodes = []):
    """Create uniform initial state over specified nodes or all nodes"""
    if nodes == []:
        state = np.ones((N,1)) / np.sqrt(N)
    else:
        state = np.zeros((N,1))
        for x in range(len(nodes)):
            state[nodes[x]] = 1
        state = state / np.sqrt(len(nodes))
    return state

def cycle_tesselation_alpha(N):
    """Create alpha tessellation pattern using NetworkX"""
    G = nx.Graph()
    
    # Add vertices (0-based indexing)
    for k in range(N):
        G.add_node(k)
    
    # Add edges for alpha pattern (even indices)
    for i in range(N-1):
        if not (i % 2):  # if i is even
            G.add_edge(i, i+1)
    
    return G

def cycle_tesselation_beta(N):
    """Create beta tessellation pattern using NetworkX"""
    G = nx.Graph()
    
    # Add vertices (0-based indexing)
    for k in range(N):
        G.add_node(k)
    
    # Add circular edge
    G.add_edge(0, N-1)
    
    # Add edges for beta pattern (odd indices)
    for i in range(N-1):
        if i % 2:  # if i is odd
            G.add_edge(i, i+1)
    
    # Add the other circular edge
    G.add_edge(N-1, 0)
    
    return G

def get_adjacency_matrix(G):
    """Get adjacency matrix from NetworkX graph with nodes sorted naturally"""
    # Sort nodes to ensure consistent ordering
    nodes = sorted(G.nodes())
    return nx.adjacency_matrix(G, nodelist=nodes).todense()

def create_noise_lists(theta, red_edge_list, blue_edge_list, deviation_range):
    """
    Create noise lists for red and blue tessellations with random deviations
    
    Parameters:
    - theta: base theta value
    - red_edge_list: list of edges in red tessellation
    - blue_edge_list: list of edges in blue tessellation
    - deviation_range: either:
        - single value (backward compatibility): random range [0, value]
        - tuple (minVal, maxVal): direct range for random.uniform(minVal, maxVal)
    
    Returns:
    - red_noise_list: list of theta values for red edges (theta + random_dev)
    - blue_noise_list: list of theta values for blue edges (theta - random_dev)
    """
    red_noise_list = []
    blue_noise_list = []
    
    # Handle different formats for deviation_range
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        # Direct (minVal, maxVal) format
        dev_min, dev_max = deviation_range
    else:
        # Backward compatibility: single value means [0, value]
        dev_min, dev_max = 0, abs(deviation_range)
    
    # Generate noise for red edges (theta + deviation)
    for _ in range(len(red_edge_list)):
        deviation = random.uniform(dev_min, dev_max)
        red_noise_list.append(theta + deviation)
    
    # Generate noise for blue edges (theta - deviation)  
    for _ in range(len(blue_edge_list)):
        deviation = random.uniform(dev_min, dev_max)
        blue_noise_list.append(theta - deviation)
    
    return red_noise_list, blue_noise_list

def create_noisy_hamiltonians(red_graph, blue_graph, red_noise_list, blue_noise_list):
    """
    Create noisy Hamiltonian matrices with individual noise parameters applied to each edge
    
    Parameters:
    - red_graph: NetworkX graph for red tessellation
    - blue_graph: NetworkX graph for blue tessellation  
    - red_noise_list: list of theta values for red edges
    - blue_noise_list: list of theta values for blue edges
    
    Returns:
    - Hr_noisy: Noisy red Hamiltonian matrix
    - Hb_noisy: Noisy blue Hamiltonian matrix
    """
    # Get base adjacency matrices
    Hr = get_adjacency_matrix(red_graph)
    Hb = get_adjacency_matrix(blue_graph)
    
    # Create noisy versions
    Hr_noisy = Hr.copy().astype(float)
    Hb_noisy = Hb.copy().astype(float)
    
    # Apply red noise parameters to red Hamiltonian
    red_edge_list = list(red_graph.edges())
    for i, edge in enumerate(red_edge_list):
        if i < len(red_noise_list):
            x, y = edge
            # Apply noise parameter to both symmetric positions
            Hr_noisy[x, y] *= red_noise_list[i]
            Hr_noisy[y, x] *= red_noise_list[i]
    
    # Apply blue noise parameters to blue Hamiltonian
    blue_edge_list = list(blue_graph.edges())
    for i, edge in enumerate(blue_edge_list):
        if i < len(blue_noise_list):
            x, y = edge
            # Apply noise parameter to both symmetric positions
            Hb_noisy[x, y] *= blue_noise_list[i]
            Hb_noisy[y, x] *= blue_noise_list[i]
    
    return Hr_noisy, Hb_noisy

def ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list):
    """Create time evolution operator with individual noise parameters for each edge"""
    # Create noisy Hamiltonians
    Hr_noisy, Hb_noisy = create_noisy_hamiltonians(red_graph, blue_graph, red_noise_list, blue_noise_list)
    
    # Create evolution operator using noisy Hamiltonians
    R = expm(1j * Hr_noisy)
    B = expm(1j * Hb_noisy)
    U = dot(B, R)
    return U, Hr_noisy, Hb_noisy

def final_state(Op, psi0, steps):
    """Evolve initial state through given number of steps"""
    psi = psi0.copy()
    for i in range(steps):
        psi = dot(Op, psi)
    return psi

def prob_vec(psiN, N):
    """Calculate probability distribution from final state"""
    probs = zeros((N, 1))
    for x in range(N):
        probs[x] = abs(psiN[x])**2
    return probs

def running(N, theta, num_steps, 
            initial_nodes=[], 
            deviation_range=0.0,
            return_all_states=False):
    """
    Run staggered quantum walk with static noise
    
    Parameters:
    - N: number of nodes in the cycle
    - theta: base theta parameter for evolution
    - num_steps: number of evolution steps
    - initial_nodes: list of initial nodes (empty list = uniform superposition)
    - deviation_range: noise deviation range - can be:
        * single value: random range [0, value]
        * tuple (max_dev, min_factor): min_dev = max_dev * min_factor, max_dev = max_dev
        * tuple (min, max): explicit range (legacy, when both > 1)
    - return_all_states: if True, return evolution states, else return final probabilities
    
    Returns:
    - If return_all_states=True: list of evolution states
    - If return_all_states=False: final probability distribution
    """
    # Create tessellations
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    # Get edge lists
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    
    # Create noise parameters
    red_noise_list, blue_noise_list = create_noise_lists(
        theta, red_edge_list, blue_edge_list, deviation_range
    )
    
    # Create initial state
    psi0 = uniform_initial_state(N, initial_nodes)
    
    # Create evolution operator
    U, Hr_noisy, Hb_noisy = ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list)
    
    if return_all_states:
        # Store all evolution states
        evolution_states = [psi0.copy()]
        psi = psi0.copy()
        
        for i in range(num_steps):
            psi = dot(U, psi)
            evolution_states.append(psi.copy())
        
        return evolution_states
    else:
        # Return only final probabilities
        psiN = final_state(U, psi0, num_steps)
        probvec = prob_vec(psiN, N)
        return probvec

def running_streaming(N, theta, num_steps, 
                     initial_nodes=[], 
                     deviation_range=0.0,
                     step_callback=None):
    """
    Run staggered quantum walk with static noise using streaming approach.
    
    This version saves memory by calling a callback function for each step
    instead of storing all states in memory.
    
    Parameters:
    - N: number of nodes in the cycle
    - theta: base theta parameter for evolution
    - num_steps: number of evolution steps
    - initial_nodes: list of initial nodes (empty list = uniform superposition)
    - deviation_range: noise deviation range - can be:
        * single value: random range [0, value]
        * tuple (minVal, maxVal): direct range for random.uniform(minVal, maxVal)
    - step_callback: function(step_idx, state) called for each step
    
    Returns:
    - final_state: the final quantum state after all steps
    """
    # Create tessellations
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    # Get edge lists
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    
    # Create noise parameters
    red_noise_list, blue_noise_list = create_noise_lists(
        theta, red_edge_list, blue_edge_list, deviation_range
    )
    
    # Create initial state
    psi0 = uniform_initial_state(N, initial_nodes)
    
    # Create evolution operator
    U, Hr_noisy, Hb_noisy = ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list)
    
    # Save initial state if callback provided
    if step_callback:
        step_callback(0, psi0.copy())
    
    # Evolve state step by step, calling callback for each step
    psi = psi0.copy()
    for i in range(num_steps):
        # Apply evolution operator
        psi_new = dot(U, psi)
        
        # Replace old state with new one to minimize memory usage
        psi = psi_new
        del psi_new  # Explicit cleanup
        
        # Call callback with current step and state copy
        if step_callback:
            step_callback(i + 1, psi.copy())
    
    return psi

def staggered_qwalk_with_noise(N, theta, steps, init_nodes=[], deviation_range=0.0):
    """
    Perform staggered quantum walk with static noise (wrapper for compatibility)
    
    Parameters:
    - N: number of nodes
    - theta: base theta parameter
    - steps: number of evolution steps
    - init_nodes: list of initial nodes (empty list = uniform superposition)
    - deviation_range: noise deviation range - can be:
        * single value: random range [0, value]
        * tuple (max_dev, min_factor): min_dev = max_dev * min_factor, max_dev = max_dev
        * tuple (min, max): explicit range (legacy, when both > 1)
    
    Returns:
    - probability distribution
    - Clean Hamiltonian matrices
    - Noisy Hamiltonian matrices
    - noise parameters
    """
    # Create tessellations
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    # Get edge lists
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    
    # Create noise parameters
    red_noise_list, blue_noise_list = create_noise_lists(
        theta, red_edge_list, blue_edge_list, deviation_range
    )
    
    # Create clean Hamiltonian matrices
    Hr_clean = get_adjacency_matrix(red_graph)
    Hb_clean = get_adjacency_matrix(blue_graph)
    
    # Create initial state
    psi0 = uniform_initial_state(N, init_nodes)
    
    # Create evolution operator and get noisy Hamiltonians
    U, Hr_noisy, Hb_noisy = ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list)
    
    # Evolve the state
    psiN = final_state(U, psi0, steps)
    
    # Calculate probabilities
    probvec = prob_vec(psiN, N)
    
    return probvec, Hr_clean, Hb_clean, Hr_noisy, Hb_noisy, red_noise_list, blue_noise_list

def print_matrix_formatted(matrix, title, precision=3):
    """Print a matrix with nice formatting"""
    print(f"\n{title}:")
    print("=" * (len(title) + 1))
    n_rows, n_cols = matrix.shape
    
    # Create format string based on precision
    fmt = f"{{:>{precision+6}.{precision}f}}"
    
    for i in range(n_rows):
        row_str = "["
        for j in range(n_cols):
            if isinstance(matrix[i, j], complex):
                real_part = matrix[i, j].real
                imag_part = matrix[i, j].imag
                if abs(imag_part) < 1e-10:  # Essentially real
                    row_str += fmt.format(real_part)
                else:
                    row_str += f"{real_part:>{precision+2}.{precision}f}{imag_part:+{precision+2}.{precision}f}j"
            else:
                row_str += fmt.format(float(matrix[i, j]))
            
            if j < n_cols - 1:
                row_str += " "
        row_str += "]"
        print(row_str)
    print()

def visualize_cycle_tessellations(N, figsize=(10, 6)):
    """Visualize both alpha and beta tessellations on the same plot with different edge colors"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Create both tessellations
    alpha_graph = cycle_tesselation_alpha(N)
    beta_graph = cycle_tesselation_beta(N)
    
    # Create a combined graph with all nodes (0-based)
    combined_graph = nx.Graph()
    for k in range(N):
        combined_graph.add_node(k)
    
    # Position nodes in a circle
    pos = {}
    for i in range(N):
        angle = 2 * np.pi * i / N
        pos[i] = (np.cos(angle), np.sin(angle))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes with labels
    nx.draw_networkx_nodes(combined_graph, pos, node_color='lightgray', 
                          node_size=800, ax=ax)
    nx.draw_networkx_labels(combined_graph, pos, font_size=16, 
                           font_weight='bold', ax=ax)
    
    # Draw alpha edges (red)
    nx.draw_networkx_edges(alpha_graph, pos, edge_color='red', 
                          width=3, alpha=0.7, ax=ax)
    
    # Draw beta edges (blue)
    nx.draw_networkx_edges(beta_graph, pos, edge_color='blue', 
                          width=3, alpha=0.7, ax=ax)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', label='Alpha tessellation')
    blue_patch = mpatches.Patch(color='blue', label='Beta tessellation')
    ax.legend(handles=[red_patch, blue_patch], loc='upper right')
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Cycle Tessellations (N={N})', fontsize=18, pad=20)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    print("Staggered Quantum Walk with Static Noise - Example")
    print("=" * 50)
    
    # Parameters
    N = 6  # Number of nodes
    theta = np.pi / 4  # Base theta parameter
    steps = 5  # Number of evolution steps
    init_nodes = [0]  # Start at node 0 only
    
    # Example 1: Old format (single value) - backward compatibility
    deviation_range_old = 0.1  # Random range [0, 0.1]
    
    # Example 2: New format (max_dev, min_factor)
    max_deviation = 0.2
    min_factor = 0.3  # minimum will be 0.2 * 0.3 = 0.06
    deviation_range_new = (max_deviation, min_factor)  # Random range [0.06, 0.2]
    
    print("Example 1: Old format (backward compatibility)")
    print(f"Parameters: N={N}, theta={theta:.3f}, steps={steps}")
    print(f"Initial nodes: {init_nodes}")
    print(f"Deviation range (old): {deviation_range_old} -> [0, {deviation_range_old}]")
    
    # Run quantum walk with old format
    probabilities_old = running(N, theta, steps, init_nodes, deviation_range_old)
    
    print("\nFinal Probability Distribution (old format):")
    print("=" * 43)
    for i in range(N):
        print(f"Node {i}: {probabilities_old[i][0]:.6f}")
    
    print(f"\nSum of probabilities: {np.sum(probabilities_old):.6f}")
    
    print("\n" + "="*50)
    print("Example 2: New format (max_dev, min_factor)")
    print(f"Parameters: N={N}, theta={theta:.3f}, steps={steps}")
    print(f"Initial nodes: {init_nodes}")
    print(f"Deviation range (new): {deviation_range_new} -> [{max_deviation * min_factor:.3f}, {max_deviation}]")
    
    # Run quantum walk with new format
    probabilities_new = running(N, theta, steps, init_nodes, deviation_range_new)
    
    print("\nFinal Probability Distribution (new format):")
    print("=" * 43)
    for i in range(N):
        print(f"Node {i}: {probabilities_new[i][0]:.6f}")
    
    print(f"\nSum of probabilities: {np.sum(probabilities_new):.6f}")
    
    # Compare with no noise
    probabilities_no_noise = running(N, theta, steps, init_nodes, deviation_range=0.0)
    
    print("\n" + "="*50)
    print("Comparison (No Noise):")
    print("=" * 22)
    for i in range(N):
        print(f"Node {i}: {probabilities_no_noise[i][0]:.6f}")
    
    print(f"\nSum of probabilities (no noise): {np.sum(probabilities_no_noise):.6f}")
    
    # Get evolution states example
    evolution_states = running(N, theta, steps, init_nodes, deviation_range_new, return_all_states=True)
    print(f"\nEvolution states collected: {len(evolution_states)} states")
    print(f"Initial state shape: {evolution_states[0].shape}")
    print(f"Final state shape: {evolution_states[-1].shape}")