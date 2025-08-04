import networkx as nx
import numpy as np

def line_tesselation_alpha(N):
    """Create alpha tessellation pattern using NetworkX"""
    G = nx.Graph()
    
    # Add vertices
    for k in range(N):
        G.add_node(k)
    
    # Add edges for alpha pattern (even indices)
    for i in range(N-1):
        if not (i % 2):  # if i is even
            G.add_edge(i, i+1)
    
    return G

def line_tesselation_beta(N):
    """Create beta tessellation pattern using NetworkX"""
    G = nx.Graph()
    
    # Add vertices
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

# Test with small N
N = 6
t1 = line_tesselation_alpha(N)
t2 = line_tesselation_beta(N)

print('Alpha tessellation edges:', list(t1.edges()))
print('Beta tessellation edges:', list(t2.edges()))

# Check adjacency matrices
A = get_adjacency_matrix(t1)
B = get_adjacency_matrix(t2)
print('Alpha adjacency matrix shape:', A.shape)
print('Beta adjacency matrix shape:', B.shape)
print('Alpha adjacency matrix:')
print(A)
print('Beta adjacency matrix:')
print(B)
