#!/usr/bin/env python3

"""
Data Regeneration Script - Test for (0, 0.8) deviation

This script processes the (0, 0.8) deviation to regenerate std and survival data,
handling missing probability distribution files gracefully.
"""

import os
import sys
import math
import numpy as np
import pickle
import argparse
from datetime import datetime

# Configuration
N = 20000
steps = N//4  # 5000
samples = 40
theta = math.pi/3

# Test only the problematic deviation
devs = [(0, 0.8)]

# Data directories
probdist_base_dir = "experiments_data_samples_probDist"
std_base_dir = "experiments_data_samples_std"
survival_base_dir = "experiments_data_samples_survivalProb"

# Survival ranges
SURVIVAL_RANGES = [
    {"name": "center_11", "from_node": "center-5", "to_node": "center+5"},
]

def format_deviation_for_filename(deviation_value):
    """Format deviation values for filename use."""
    if isinstance(deviation_value, (tuple, list)) and len(deviation_value) == 2:
        min_val, max_val = deviation_value
        return f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        return f"{deviation_value:.3f}"

def resolve_node_position(node_spec, N):
    """Resolve node position specifications."""
    center = N // 2
    
    if isinstance(node_spec, int):
        return node_spec
    elif isinstance(node_spec, str):
        if node_spec == "center":
            return center
        elif node_spec.startswith("center"):
            if "+" in node_spec:
                offset = int(node_spec.split("+")[1])
                return center + offset
            elif "-" in node_spec:
                offset = int(node_spec.split("-")[1])
                return center - offset
        else:
            return int(node_spec)
    else:
        raise ValueError(f"Unknown node specification: {node_spec}")

def calculate_std_from_probdist(probdist, N):
    """Calculate standard deviation from probability distribution."""
    center = N // 2
    positions = np.arange(N) - center
    mean_pos = np.sum(positions * probdist)
    variance = np.sum(((positions - mean_pos) ** 2) * probdist)
    return np.sqrt(variance)

def calculate_survival_probability_from_array(prob_array, from_node, to_node):
    """Calculate survival probability as sum of probabilities in range."""
    if from_node == to_node:
        return float(prob_array[from_node])
    else:
        return float(np.sum(prob_array[from_node:to_node+1]))

def load_probdist_file(filepath):
    """Load a probability distribution file safely."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        return None

def main():
    dev = devs[0]
    print(f"Processing deviation: {dev}")
    
    # Format deviation for directory name
    dev_dir = f"dev_{format_deviation_for_filename(dev)}"
    
    # Construct source path
    base_path = os.path.join(probdist_base_dir, "dummy_tesselation_func_static_noise",
                            f"theta_{theta:.6f}", dev_dir, f"N_{N}")
    samples_dir = os.path.join(base_path, f"samples_{samples}")
    
    print(f"Source directory: {samples_dir}")
    
    if not os.path.exists(samples_dir):
        print("Source directory not found!")
        return
    
    # Count available files
    available_files = 0
    for step in range(steps):
        probdist_file = os.path.join(samples_dir, f"mean_step_{step}.pkl")
        if os.path.exists(probdist_file):
            available_files += 1
    
    print(f"Available files: {available_files}/{steps}")
    
    # Process std data
    print("Processing standard deviation data...")
    std_values = []
    
    for step in range(steps):
        probdist_file = os.path.join(samples_dir, f"mean_step_{step}.pkl")
        
        if not os.path.exists(probdist_file):
            std_values.append(None)
            continue
        
        probdist = load_probdist_file(probdist_file)
        if probdist is None:
            std_values.append(None)
            continue
        
        std_val = calculate_std_from_probdist(probdist, N)
        std_values.append(std_val)
        
        if step % 500 == 0:
            print(f"  Processed step {step}: std = {std_val:.6f}")
    
    # Save std data
    std_output_dir = os.path.join(std_base_dir, "dummy_tesselation_func_static_noise", 
                                 f"theta_{theta:.6f}", dev_dir, f"N_{N}", f"samples_{samples}")
    os.makedirs(std_output_dir, exist_ok=True)
    
    std_output_file = os.path.join(std_output_dir, "std_vs_time.pkl")
    with open(std_output_file, 'wb') as f:
        pickle.dump(std_values, f)
    
    print(f"STD data saved to: {std_output_file}")
    
    # Process survival data
    print("Processing survival probability data...")
    
    for range_config in SURVIVAL_RANGES:
        range_name = range_config["name"]
        from_node = resolve_node_position(range_config["from_node"], N)
        to_node = resolve_node_position(range_config["to_node"], N)
        
        print(f"  Range '{range_name}': nodes {from_node} to {to_node}")
        
        survival_values = []
        
        for step in range(steps):
            probdist_file = os.path.join(samples_dir, f"mean_step_{step}.pkl")
            
            if not os.path.exists(probdist_file):
                survival_values.append(None)
                continue
            
            probdist = load_probdist_file(probdist_file)
            if probdist is None:
                survival_values.append(None)
                continue
            
            survival_prob = calculate_survival_probability_from_array(probdist, from_node, to_node)
            survival_values.append(survival_prob)
            
            if step % 500 == 0:
                print(f"    Processed step {step}: survival = {survival_prob:.6f}")
        
        # Save survival data
        survival_output_dir = os.path.join(survival_base_dir, "dummy_tesselation_func_static_noise",
                                         f"theta_{theta:.6f}", dev_dir, f"N_{N}", f"samples_{samples}")
        os.makedirs(survival_output_dir, exist_ok=True)
        
        survival_output_file = os.path.join(survival_output_dir, f"survival_{range_name}.pkl")
        with open(survival_output_file, 'wb') as f:
            pickle.dump(survival_values, f)
        
        print(f"  Survival data saved to: {survival_output_file}")
    
    print("Processing completed!")

if __name__ == "__main__":
    main()
