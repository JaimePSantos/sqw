#!/usr/bin/env python3

"""
Barebones Staggered Quantum Walk Test Script

This script implements the basic staggered quantum walk calculation to verify:
1. How the noise is applied
2. Whether theta = pi vs theta = pi/3 affects the standard deviation
3. Why the (0,0) noiseless case has such small values
"""

import numpy as np
import math
from scipy.linalg import expm
import random

def create_cycle_tessellation_alpha(N):
    """Create alpha tessellation (even pairs)"""
    edges = []
    for i in range(0, N, 2):
        edges.append((i, (i + 1) % N))
    return edges

def create_cycle_tessellation_beta(N):
    """Create beta tessellation (odd pairs)"""
    edges = []
    for i in range(1, N, 2):
        edges.append((i, (i + 1) % N))
    return edges

def create_adjacency_matrix(N, edges):
    """Create adjacency matrix from edge list"""
    A = np.zeros((N, N))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A

def create_noise_list(theta, num_edges, deviation_range):
    """Create noise list for edges"""
    noise_list = []
    
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        dev_min, dev_max = deviation_range
    else:
        dev_min, dev_max = 0, abs(deviation_range)
    
    for _ in range(num_edges):
        deviation = random.uniform(dev_min, dev_max)
        noise_list.append(theta + deviation)
    
    return noise_list

def create_evolution_operator(N, theta, deviation_range=(0, 0), random_seed=42):
    """Create evolution operator U = exp(i*theta*H_beta) * exp(i*theta*H_alpha)"""
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create tessellations
    alpha_edges = create_cycle_tessellation_alpha(N)
    beta_edges = create_cycle_tessellation_beta(N)
    
    # Create adjacency matrices
    H_alpha = create_adjacency_matrix(N, alpha_edges)
    H_beta = create_adjacency_matrix(N, beta_edges)
    
    # For noiseless case, use base theta directly
    if deviation_range == (0, 0) or (isinstance(deviation_range, (tuple, list)) and 
                                      len(deviation_range) == 2 and 
                                      deviation_range[0] == 0 and deviation_range[1] == 0):
        # No noise - use clean theta everywhere
        U_alpha = expm(1j * theta * H_alpha)
        U_beta = expm(1j * theta * H_beta)
        print(f"[NOISELESS] Using theta = {theta:.6f} for all edges")
    else:
        # With noise - this is simplified, normally each edge would get individual noise
        alpha_noise = create_noise_list(theta, len(alpha_edges), deviation_range)
        beta_noise = create_noise_list(theta, len(beta_edges), deviation_range)
        
        # For simplicity, use average of noise values (real implementation varies per edge)
        avg_alpha_theta = np.mean(alpha_noise)
        avg_beta_theta = np.mean(beta_noise)
        
        U_alpha = expm(1j * avg_alpha_theta * H_alpha)
        U_beta = expm(1j * avg_beta_theta * H_beta)
        print(f"[NOISE] Alpha avg theta = {avg_alpha_theta:.6f}, Beta avg theta = {avg_beta_theta:.6f}")
    
    # Evolution operator: U = U_beta * U_alpha
    U = np.dot(U_beta, U_alpha)
    
    return U, H_alpha, H_beta

def create_initial_state(N, center_node=None):
    """Create initial state - localized at center if specified, uniform otherwise"""
    psi0 = np.zeros((N, 1), dtype=complex)
    
    if center_node is not None:
        psi0[center_node] = 1.0
    else:
        # Uniform superposition
        psi0[:, 0] = 1.0 / np.sqrt(N)
    
    return psi0

def evolve_state(U, psi0, steps):
    """Evolve state through multiple steps"""
    psi = psi0.copy()
    for step in range(steps):
        psi = np.dot(U, psi)
    return psi

def calculate_probability_distribution(psi, N):
    """Calculate probability distribution from state"""
    prob = np.zeros(N)
    for i in range(N):
        prob[i] = abs(psi[i, 0])**2
    return prob

def calculate_standard_deviation(prob, N):
    """Calculate standard deviation of probability distribution"""
    # Create position array (centered at N//2)
    positions = np.arange(N) - N // 2
    
    # Calculate mean position
    mean_pos = np.sum(positions * prob)
    
    # Calculate variance
    variance = np.sum(((positions - mean_pos)**2) * prob)
    
    # Standard deviation
    std_dev = np.sqrt(variance)
    
    return std_dev

def run_multiple_samples(N, theta, steps, deviation_range, num_samples=5):
    """Run multiple samples and calculate standard deviation across samples"""
    
    print(f"\n=== Testing: N={N}, theta={theta:.6f}, steps={steps}, dev={deviation_range}, samples={num_samples} ===")
    
    # Store final probability distributions for each sample
    prob_distributions = []
    std_devs_individual = []
    
    for sample in range(num_samples):
        # Create evolution operator (different random seed for each sample)
        U, H_alpha, H_beta = create_evolution_operator(N, theta, deviation_range, random_seed=42+sample)
        
        # Create initial state (localized at center)
        center_node = N // 2
        psi0 = create_initial_state(N, center_node)
        
        # Evolve state
        psi_final = evolve_state(U, psi0, steps)
        
        # Calculate probability distribution
        prob = calculate_probability_distribution(psi_final, N)
        prob_distributions.append(prob)
        
        # Calculate individual standard deviation
        std_dev = calculate_standard_deviation(prob, N)
        std_devs_individual.append(std_dev)
        
        print(f"  Sample {sample+1}: std_dev = {std_dev:.6e}")
    
    # Calculate standard deviation across samples (this is what we plot)
    prob_distributions = np.array(prob_distributions)
    
    # Calculate variance across samples for each position
    variances_across_samples = np.var(prob_distributions, axis=0)
    
    # Total standard deviation across samples
    total_variance = np.sum(variances_across_samples)
    total_std_dev = np.sqrt(total_variance)
    
    print(f"  Individual std devs: {std_devs_individual}")
    print(f"  Total std dev across samples: {total_std_dev:.6e}")
    print(f"  Max variance across samples: {np.max(variances_across_samples):.6e}")
    print(f"  Sum of variances: {total_variance:.6e}")
    
    return total_std_dev, std_devs_individual, prob_distributions

def test_different_theta_values():
    """Test how different theta values affect the noiseless case"""
    
    N = 100  # Smaller system for quick testing
    steps = 25
    num_samples = 5
    
    theta_values = [
        math.pi / 3,    # Original from your diagnostics  
        math.pi,        # Current experiments
        math.pi / 2,    # Quarter turn
        math.pi / 4,    # Eighth turn
    ]
    
    deviation_ranges = [
        (0, 0),     # Noiseless
        (0, 0.2),   # Small noise
    ]
    
    print("THETA VALUE COMPARISON:")
    print("=" * 60)
    
    for dev_range in deviation_ranges:
        print(f"\nDeviation range: {dev_range}")
        print("-" * 40)
        
        for theta in theta_values:
            std_dev, individual_stds, prob_dists = run_multiple_samples(
                N, theta, steps, dev_range, num_samples
            )
            
            print(f"theta = {theta:.6f} ({theta/math.pi:.3f}π): total_std = {std_dev:.6e}")

if __name__ == "__main__":
    test_different_theta_values()
