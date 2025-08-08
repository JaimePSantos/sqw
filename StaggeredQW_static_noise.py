#StaggeredQuantumWalk

from numpy import *
from matplotlib.pyplot import *
import networkx as nx
from scipy import linalg
import matplotlib.patches as mpatches
from fractions import Fraction
import random

rcParams['figure.figsize'] = 11, 8
matplotlib.rcParams.update({'font.size': 15})
def visualize_cycle_tessellations(N, figsize=(10, 6)):
    """Visualize both alpha and beta tessellations on the same plot with different edge colors"""
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
        angle = 2 * pi * i / N
        pos[i] = (cos(angle), sin(angle))
    
    # Create the plot
    fig, ax = subplots(figsize=figsize)
    
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
    
    tight_layout()
    show()

def uniform_initial_state(N, nodes = []):
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

red_graph = cycle_tesselation_alpha(4)
blue_graph = cycle_tesselation_beta(4)

red_tesselation = []
for edge in red_graph.edges():
    red_tesselation.append(list(edge))

blue_tesselation = []
for edge in blue_graph.edges():
    blue_tesselation.append(list(edge))

print("Red tessellation:", red_tesselation)
print("Blue tessellation:", blue_tesselation)


# Example usage
visualize_cycle_tessellations(4)

def create_hamiltonian(red_graph, blue_graph, red_graph_noise=[], blue_graph_noise=[]):
    """
    Create Hamiltonian matrix from tessellation graphs with noise parameters
    
    Parameters:
    - red_graph: NetworkX graph for alpha tessellation
    - blue_graph: NetworkX graph for beta tessellation  
    - red_graph_noise: list of theta values for red edges [theta_red_1, theta_red_2, ...]
    - blue_graph_noise: list of theta values for blue edges [theta_blue_1, theta_blue_2, ...]
    
    Returns:
    - Hamiltonian matrix (numpy array)
    """
    # Get edge lists
    red_edge_list = list(red_graph.edges())
    blue_edge_list = list(blue_graph.edges())
    
    # Create combined graph with all nodes
    combined_graph = nx.Graph()
    combined_graph.add_nodes_from(red_graph.nodes())
    combined_graph.add_nodes_from(blue_graph.nodes())
    combined_graph.add_edges_from(red_edge_list)
    combined_graph.add_edges_from(blue_edge_list)
    
    # Get adjacency matrix
    node_list = sorted(combined_graph.nodes())
    adj_matrix = nx.adjacency_matrix(combined_graph, nodelist=node_list).toarray().astype(float)
    
    # Apply red noise parameters
    if red_graph_noise:
        for i, edge in enumerate(red_edge_list):
            if i < len(red_graph_noise):
                x, y = edge
                # Apply theta to both symmetric positions (already 0-based)
                adj_matrix[x][y] *= red_graph_noise[i]
                adj_matrix[y][x] *= red_graph_noise[i]
    
    # Apply blue noise parameters
    if blue_graph_noise:
        for i, edge in enumerate(blue_edge_list):
            if i < len(blue_graph_noise):
                x, y = edge
                # Apply theta to both symmetric positions (already 0-based)
                adj_matrix[x][y] *= blue_graph_noise[i]
                adj_matrix[y][x] *= blue_graph_noise[i]
    
    return adj_matrix

def create_noise_lists(theta, red_edge_list, blue_edge_list, deviation_range):
    """
    Create noise lists for red and blue tessellations with random deviations
    
    Parameters:
    - theta: base theta value
    - red_edge_list: list of edges in red tessellation
    - blue_edge_list: list of edges in blue tessellation
    - deviation_range: maximum deviation value (applied as +/- range)
    
    Returns:
    - red_noise_list: list of theta values for red edges (theta + random_dev)
    - blue_noise_list: list of theta values for blue edges (theta - random_dev)
    """
    red_noise_list = []
    blue_noise_list = []
    
    # Generate noise for red edges (theta + deviation)
    for _ in range(len(red_edge_list)):
        deviation = random.uniform(-deviation_range, deviation_range)
        red_noise_list.append(theta + deviation)
    
    # Generate noise for blue edges (theta - deviation)
    for _ in range(len(blue_edge_list)):
        deviation = random.uniform(-deviation_range, deviation_range)
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

def get_adjacency_matrix(G):
    """Get adjacency matrix from NetworkX graph with nodes sorted naturally"""
    # Sort nodes to ensure consistent ordering
    nodes = sorted(G.nodes())
    return nx.adjacency_matrix(G, nodelist=nodes).todense()

def ct_evo(Hr, Hb, theta_r, theta_b):
    """Create time evolution operator for staggered quantum walk"""
    R = linalg.expm(1j * theta_r * Hr)
    B = linalg.expm(1j * theta_b * Hb)
    U = dot(B, R)
    return U

def ct_evo_with_noise(red_graph, blue_graph, red_noise_list, blue_noise_list):
    """Create time evolution operator with individual noise parameters for each edge"""
    # Create noisy Hamiltonians
    Hr_noisy, Hb_noisy = create_noisy_hamiltonians(red_graph, blue_graph, red_noise_list, blue_noise_list)
    
    # Create evolution operator using noisy Hamiltonians
    R = linalg.expm(1j * Hr_noisy)
    B = linalg.expm(1j * Hb_noisy)
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

def staggered_qwalk_with_noise(N, theta, steps, init_nodes=[], deviation_range=0.0):
    """
    Perform staggered quantum walk with static noise
    
    Parameters:
    - N: number of nodes
    - theta: base theta parameter
    - steps: number of evolution steps
    - init_nodes: list of initial nodes (empty list = uniform superposition)
    - deviation_range: noise deviation range
    
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
    
    # Create initial state using your uniform_initial_state function
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

# Usage example for N = 4
if __name__ == "__main__":
    print("Staggered Quantum Walk with Static Noise - Example for N=4")
    print("=" * 60)
    
    N = 4
    theta = pi / 4
    steps = 10
    init_nodes = [0]  # Start at node 0 only
    deviation_range = 0.1  # Small noise deviation
    
    # Run the quantum walk
    probabilities, Hr_clean, Hb_clean, Hr_noisy, Hb_noisy, red_noise, blue_noise = staggered_qwalk_with_noise(
        N, theta, steps, init_nodes, deviation_range
    )
    
    # Print the tessellation information
    red_graph = cycle_tesselation_alpha(N)
    blue_graph = cycle_tesselation_beta(N)
    
    print(f"Red tessellation edges: {list(red_graph.edges())}")
    print(f"Blue tessellation edges: {list(blue_graph.edges())}")
    print(f"Base theta: {theta:.3f}")
    print(f"Red noise parameters: {[f'{x:.3f}' for x in red_noise]}")
    print(f"Blue noise parameters: {[f'{x:.3f}' for x in blue_noise]}")
    print(f"Initial state: localized at node {init_nodes}")
    
    # Print clean Hamiltonian matrices
    print_matrix_formatted(Hr_clean, "Clean Red Hamiltonian (Hr)")
    print_matrix_formatted(Hb_clean, "Clean Blue Hamiltonian (Hb)")
    
    # Print noisy Hamiltonian matrices
    print_matrix_formatted(Hr_noisy, "NOISY Red Hamiltonian (Hr_noisy)")
    print_matrix_formatted(Hb_noisy, "NOISY Blue Hamiltonian (Hb_noisy)")
    
    # Print final probabilities
    print("Final Probability Distribution:")
    print("=" * 32)
    for i in range(N):
        print(f"Node {i}: {probabilities[i][0]:.6f}")
    
    print(f"\nSum of probabilities: {np.sum(probabilities):.6f}")
    
    # Also show uniform superposition example
    print("\n" + "="*60)
    print("Example: Uniform Superposition Initial State")
    print("="*60)
    
    probabilities_uniform, _, _, Hr_noisy_uni, Hb_noisy_uni, red_noise_uni, blue_noise_uni = staggered_qwalk_with_noise(
        N, theta, steps, [], deviation_range  # Empty list = uniform superposition
    )
    
    print(f"Red noise parameters: {[f'{x:.3f}' for x in red_noise_uni]}")
    print(f"Blue noise parameters: {[f'{x:.3f}' for x in blue_noise_uni]}")
    print("Initial state: uniform superposition over all nodes")
    
    print_matrix_formatted(Hr_noisy_uni, "NOISY Red Hamiltonian (Uniform case)")
    print_matrix_formatted(Hb_noisy_uni, "NOISY Blue Hamiltonian (Uniform case)")
    
    print("Final Probability Distribution (Uniform Start):")
    print("=" * 47)
    for i in range(N):
        print(f"Node {i}: {probabilities_uniform[i][0]:.6f}")
    
    print(f"\nSum of probabilities: {np.sum(probabilities_uniform):.6f}")
    
    # Also run without noise for comparison
    print("\n" + "="*60)
    print("Comparison: Same walk WITHOUT noise (localized start)")
    print("="*60)
    
    probabilities_no_noise, Hr_clean_no_noise, Hb_clean_no_noise, _, _, _, _ = staggered_qwalk_with_noise(
        N, theta, steps, [0], deviation_range=0.0
    )
    
    print_matrix_formatted(Hr_clean_no_noise, "Clean Red Hamiltonian (No Noise)")
    print_matrix_formatted(Hb_clean_no_noise, "Clean Blue Hamiltonian (No Noise)")
    print("Initial state: localized at node [0]")
    
    print("Final Probability Distribution (No Noise):")
    print("=" * 42)
    for i in range(N):
        print(f"Node {i}: {probabilities_no_noise[i][0]:.6f}")
    
    print(f"\nSum of probabilities: {np.sum(probabilities_no_noise):.6f}")