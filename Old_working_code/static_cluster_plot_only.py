#!/usr/bin/env python3

"""
Static Noise Experiment - Plotting Only Module (Optimized)

This script loads existing probability distribution and standard deviation data
and creates the plots defined in the main static cluster experiment.

OPTIMIZATIONS:
- Only loads final step probability distributions for final distribution plotting
- Directly loads pre-computed standard deviation data from files
- Significantly faster than loading all time steps for plotting

It extracts all necessary data loading functions and plotting routines
from static_cluster_logged_mp.py to provide a standalone plotting capability.

Usage:
    python static_cluster_plot_only.py
    
Configuration:
- Modify the plotting parameters and deviation values below
- Ensure the experiments_data folders exist with the required data
"""

import time
import math
import numpy as np
import os
import sys
import pickle
from datetime import datetime

# Try to import smart loading functions for probability distributions
try:
    from smart_loading_static import load_mean_probability_distributions
    SMART_LOADING_AVAILABLE = True
    print("[OK] Smart loading functions available for survival probability generation")
except ImportError as e:
    SMART_LOADING_AVAILABLE = False
    print(f"[WARNING] Smart loading functions not available: {e}")
    print("[INFO] Survival probability generation will be disabled if data doesn't exist")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Global plotting control
ENABLE_PLOTTING = True  # Master switch for all plotting

# Survival probability generation control
GENERATE_SURVIVAL_PROB_IF_MISSING = True  # Generate survival prob data if it doesn't exist
SAVE_GENERATED_SURVIVAL_PROB = True       # Save generated survival prob data for future use

# Standard Deviation vs Time Plot Configuration
STD_PLOT_CONFIG = {
    'enabled': True,                    # Enable/disable this specific plot
    'use_loglog': True,                 # Use log-log scale instead of linear
    'figure_size': (12, 8),            # Figure size in inches (width, height)
    'save_figure': True,                # Save plot to file
    'filename_linear': 'static_noise_std_vs_time_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'filename_loglog': 'static_noise_std_vs_time_loglog_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'title_linear': 'Standard Deviation vs Time for Different Static Noise Deviations',
    'title_loglog': 'Standard Deviation vs Time (Log-Log Scale) for Different Static Noise Deviations',
    'xlabel': 'Time Step',
    'ylabel': 'Standard Deviation',
    'fontsize_title': 14,
    'fontsize_labels': 12,
    'fontsize_legend': 10,
    'linewidth': 2,
    'markersize': 0,
    'marker_circle': 'o',
    'marker_square': 's',
    'marker_square_size': 4,
    'linestyle_noiseless': '--',
    'alpha_noiseless': 0.8,
    'alpha_regular': 1.0,
    'grid_alpha': 0.3,
    'grid_which': 'both',               # For log-log: 'both', 'major', 'minor'
    'grid_linestyle': '-',
    'epsilon_noiseless': 1e-3,          # Small value for noiseless case in log-log
    'dpi': 300,                         # DPI for saved figures
    'bbox_inches': 'tight'              # Bbox setting for saved figures
}

# Final Probability Distribution Plot Configuration  
PROBDIST_PLOT_CONFIG = {
    'enabled': True,                    # Enable/disable this specific plot
    'figure_size': (14, 8),            # Figure size in inches (width, height)
    'save_figure': True,                # Save plot to file
    'filename': 'static_noise_final_probdist_log_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'title': 'Probability Distributions at Final Time Step (t={final_step}) - Log Scale',
    'xlabel': 'Position',
    'ylabel': 'Probability (log scale)',
    'fontsize_title': 14,
    'fontsize_labels': 12,
    'fontsize_legend': 10,
    'linewidth': 2,
    'alpha': 0.8,
    'xlim': (-80, 80),               # X-axis limits
    'ylim_min': 1e-5,                  # Y-axis minimum (None for auto)
    'ylim_max': 1e0,                  # Y-axis maximum (None for auto)
    'grid_alpha': 0.3,
    'dpi': 300,                        # DPI for saved figures
    'bbox_inches': 'tight'             # Bbox setting for saved figures
}

# Survival Probability Plot Configuration
SURVIVAL_PLOT_CONFIG = {
    'enabled': True,                    # Enable/disable this specific plot
    'use_loglog': True,               # Use log-log scale instead of linear
    'use_semilogy': True,              # Use log-linear (semilogy) scale
    'figure_size': (12, 8),            # Figure size in inches (width, height)
    'save_figure': True,                # Save plot to file
    'filename_linear': 'static_noise_survival_prob_linear_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'filename_loglog': 'static_noise_survival_prob_loglog_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'filename_semilogy': 'static_noise_survival_prob_semilogy_N{N}_steps{steps}_samples{samples}_theta{theta:.3f}.png',
    'title_linear': 'Survival Probability vs Time for Different Static Noise Deviations',
    'title_loglog': 'Survival Probability vs Time (Log-Log Scale) for Different Static Noise Deviations',
    'title_semilogy': 'Survival Probability vs Time (Semi-Log Scale) for Different Static Noise Deviations',
    'xlabel': 'Time Step',
    'ylabel_linear': 'Survival Probability',
    'ylabel_loglog': 'Survival Probability (log scale)',
    'ylabel_semilogy': 'Survival Probability (log scale)',
    'fontsize_title': 14,
    'fontsize_labels': 12,
    'fontsize_legend': 10,
    'linewidth': 2,
    'markersize': 0,
    'alpha_regular': 1.0,
    'grid_alpha': 0.3,
    'grid_which': 'both',               # For log-log: 'both', 'major', 'minor'
    'grid_linestyle': '-',
    'dpi': 300,                         # DPI for saved figures
    'bbox_inches': 'tight',             # Bbox setting for saved figures
    'survival_range': 'center_11'       # Default survival range to plot
}

# Survival probability ranges configuration
SURVIVAL_RANGES = [
    {"name": "center_single", "from_node": "center", "to_node": "center"},  # Single center node
    {"name": "center_5", "from_node": "center-2", "to_node": "center+2"},   # 5 nodes around center
    {"name": "center_11", "from_node": "center-5", "to_node": "center+5"},  # 11 nodes around center
    {"name": "center_21", "from_node": "center-10", "to_node": "center+10"}, # 21 nodes around center
]

# Label formatting configuration
LABEL_CONFIG = {
    'precision_new_format': 3,          # Decimal places for new format labels
    'precision_legacy_format': 3,       # Decimal places for legacy format labels
    'precision_single_value': 3,        # Decimal places for single value labels
    'new_format_template': 'max{max_dev:.{prec}f}_min{min_val:.{prec}f}',
    'legacy_format_template': 'min{min_val:.{prec}f}_max{max_val:.{prec}f}',
    'single_value_template': '{dev:.{prec}f}',
    'plot_label_template': 'Static deviation = {dev_label}'
}

# Experiment parameters (must match the original experiment)
N = 300  # System size
steps = N//4  # Time steps
samples = 5  # Samples per deviation

# Quantum walk parameters
theta = math.pi/3  # Base theta parameter for static noise
initial_state_kwargs = {"nodes": [N//2]}

# Deviation values for static noise experiments
# MUST MATCH the original experiment configuration
# devs = [
#     (0,0),              # No noise
#     (0, 0.2),           # Small noise range
#     (0, 0.6),           # Medium noise range  
#     (0, 0.8),           # Medium noise range  
#     (0, 1),           # Medium noise range  
# ]

devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.5),           # Medium noise range  
]

# Data directories
probdist_base_dir = "experiments_data_samples_probDist"
std_base_dir = "experiments_data_samples_std"
survival_base_dir = "experiments_data_samples_survivalProb"
noise_type = "static_noise"

# ============================================================================
# LOADING FUNCTIONS (extracted from smart_loading_static.py)
# ============================================================================

def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

def format_deviation_for_filename(deviation_value, use_legacy_format=None):
    """
    Format deviation values for filename use.
    Handles both single values and tuple ranges.
    """
    if isinstance(deviation_value, (tuple, list)) and len(deviation_value) == 2:
        # Tuple format (min, max)
        min_val, max_val = deviation_value
        if use_legacy_format is False:
            # New format: max_dev and min_factor
            return f"max{max_val:.6f}_min{min_val:.6f}"
        else:
            # Legacy format: min and max
            return f"min{min_val:.6f}_max{max_val:.6f}"
    else:
        # Single value format
        return f"{deviation_value:.6f}"

def get_experiment_dir(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data_samples", theta=None):
    """
    Construct experiment directory path following the expected structure.
    """
    tesselation_name = tesselation_func.__name__
    
    if noise_type == "static_noise":
        folder = f"{tesselation_name}_static_noise"
        base = os.path.join(base_dir, folder)
        
        if theta is not None:
            theta_folder = f"theta_{theta:.6f}"
            base = os.path.join(base, theta_folder)
        
        if has_noise and noise_params is not None:
            dev_folder = f"dev_{format_deviation_for_filename(noise_params[0])}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")
    else:
        # Other noise types
        noise_str = f"{noise_type}" if has_noise else f"{noise_type}_nonoise"
        folder = f"{tesselation_name}_{noise_str}"
        base = os.path.join(base_dir, folder)
        
        if theta is not None:
            theta_folder = f"theta_{theta:.6f}"
            base = os.path.join(base, theta_folder)
        
        if has_noise and noise_params is not None:
            dev_folder = f"dev_{format_deviation_for_filename(noise_params[0])}"
            return os.path.join(base, dev_folder, f"N_{N}")
        else:
            return os.path.join(base, f"N_{N}")

def find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=None, noise_type="static_noise", base_dir="experiments_data_samples", theta=None):
    """
    Find experiment directory supporting both old and new formats.
    Returns: Tuple (directory_path, found_format)
    """
    if noise_type != "static_noise" or not noise_params:
        return get_experiment_dir(tesselation_func, has_noise, N, noise_params, noise_type, base_dir, theta), 'unified'
    
    deviation_range = noise_params[0] if noise_params else 0
    tesselation_name = tesselation_func.__name__
    
    # Try legacy format first (since that's what we have)
    legacy_folder = f"{tesselation_name}_static_noise"
    legacy_base = os.path.join(base_dir, legacy_folder)
    
    if theta is not None:
        theta_folder = f"theta_{theta:.6f}"
        legacy_base = os.path.join(legacy_base, theta_folder)
    
    # For legacy format, use the old naming convention
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        min_val, max_val = deviation_range
        legacy_dev_suffix = f"min{min_val:.3f}_max{max_val:.3f}"
    else:
        legacy_dev_suffix = f"{deviation_range:.6f}"
    
    legacy_dev_folder = f"dev_{legacy_dev_suffix}"
    legacy_path = os.path.join(legacy_base, legacy_dev_folder, f"N_{N}")
    
    if os.path.exists(legacy_path):
        return legacy_path, 'legacy'
    
    # Try unified structure as fallback
    if isinstance(deviation_range, (tuple, list)) and len(deviation_range) == 2:
        unified_dev_suffix = format_deviation_for_filename(deviation_range, use_legacy_format=False)
    else:
        unified_dev_suffix = format_deviation_for_filename(deviation_range, use_legacy_format=True)
    
    unified_dev_folder = f"dev_{unified_dev_suffix}"
    unified_path = os.path.join(legacy_base, unified_dev_folder, f"N_{N}")
    
    if os.path.exists(unified_path):
        return unified_path, 'unified'
    
    # Return legacy path even if it doesn't exist (for consistency)
    return legacy_path, 'legacy'

def load_final_step_probability_distributions(tesselation_func, N, steps, parameter_list, probdist_base_dir, noise_type, theta):
    """
    Load only the final step probability distributions from the probDist folder.
    Optimized for final distribution plotting.
    """
    final_step = steps - 1
    print(f"Loading final step probability distributions (step {final_step}) for {len(parameter_list)} deviations...")
    
    results = []
    for i, params in enumerate(parameter_list):
        exp_dir, format_used = find_experiment_dir_flexible(
            tesselation_func, True, N, [params], noise_type, probdist_base_dir, theta
        )
        
        # Check if the experiment directory exists
        if not os.path.exists(exp_dir):
            print(f"  [MISSING] Experiment directory not found: {exp_dir}")
            results.append(None)
            continue
        
        print(f"  [LOADING] Dev {i+1}/{len(parameter_list)}: {params} from {exp_dir}")
        
        try:
            # First, check if files are directly in the experiment directory
            prob_file_new = os.path.join(exp_dir, f"mean_step_{final_step}.pkl")
            prob_file_old = os.path.join(exp_dir, f"probDist_step_{final_step}.pkl")
            
            prob_file = None
            if os.path.exists(prob_file_new):
                prob_file = prob_file_new
            elif os.path.exists(prob_file_old):
                prob_file = prob_file_old
            else:
                # If not found directly, check inside samples subfolder
                samples_folder = f"samples_{samples}"
                samples_dir = os.path.join(exp_dir, samples_folder)
                
                if os.path.exists(samples_dir):
                    prob_file_new_samples = os.path.join(samples_dir, f"mean_step_{final_step}.pkl")
                    prob_file_old_samples = os.path.join(samples_dir, f"probDist_step_{final_step}.pkl")
                    
                    if os.path.exists(prob_file_new_samples):
                        prob_file = prob_file_new_samples
                    elif os.path.exists(prob_file_old_samples):
                        prob_file = prob_file_old_samples
            
            if prob_file and os.path.exists(prob_file):
                with open(prob_file, 'rb') as f:
                    prob_dist = pickle.load(f)
                    results.append(prob_dist)
                    print(f"  [OK] Loaded final step probability distribution from {os.path.relpath(prob_file, exp_dir)}")
            else:
                print(f"  [MISSING] Final step file not found in {exp_dir} or {exp_dir}/samples_{samples}")
                results.append(None)
            
        except Exception as e:
            print(f"  [ERROR] Failed to load dev {params}: {e}")
            results.append(None)
    
    return results

def load_std_data_directly(devs, N, tesselation_func, std_base_dir, noise_type, theta=None):
    """
    Load standard deviation data directly from std files.
    Optimized for standard deviation plotting.
    """
    print(f"\n[DATA] Loading standard deviation data directly from '{std_base_dir}'...")
    
    stds = []
    
    for i, dev in enumerate(devs):
        print(f"\n[STD {i+1}/{len(devs)}] Loading deviation: {dev}")
        
        # Get experiment directory for this deviation
        exp_dir, format_used = find_experiment_dir_flexible(
            tesselation_func, True, N, [dev], noise_type, std_base_dir, theta
        )
        
        # First, try to find std files directly in experiment directory
        std_files = [
            os.path.join(exp_dir, "std_data.pkl"),
            os.path.join(exp_dir, "std_vs_time.pkl")
        ]
        
        # Try to load existing std data
        loaded = False
        for std_file in std_files:
            try:
                if os.path.exists(std_file):
                    with open(std_file, 'rb') as f:
                        std_values = pickle.load(f)
                    print(f"  [OK] Loaded {len(std_values)} std values from {os.path.relpath(std_file, exp_dir)}")
                    stds.append(std_values)
                    loaded = True
                    break
            except Exception as e:
                print(f"  [WARNING] Could not load std data from {std_file}: {e}")
                continue
        
        # If not found directly, check inside samples subfolder
        if not loaded:
            samples_folder = f"samples_{samples}"
            samples_dir = os.path.join(exp_dir, samples_folder)
            
            if os.path.exists(samples_dir):
                std_files_samples = [
                    os.path.join(samples_dir, "std_data.pkl"),
                    os.path.join(samples_dir, "std_vs_time.pkl")
                ]
                
                for std_file in std_files_samples:
                    try:
                        if os.path.exists(std_file):
                            with open(std_file, 'rb') as f:
                                std_values = pickle.load(f)
                            print(f"  [OK] Loaded {len(std_values)} std values from {os.path.relpath(std_file, exp_dir)}")
                            stds.append(std_values)
                            loaded = True
                            break
                    except Exception as e:
                        print(f"  [WARNING] Could not load std data from {std_file}: {e}")
                        continue
        
        if not loaded:
            print(f"  [MISSING] Standard deviation file not found in: {exp_dir}")
            print(f"  [MISSING] Also checked samples subfolder: {os.path.join(exp_dir, f'samples_{samples}')}")
            stds.append([])
    
    print(f"[OK] Standard deviation data loading completed!")
    return stds

# ============================================================================
# SURVIVAL PROBABILITY LOADING FUNCTIONS
# ============================================================================

def resolve_node_position(node_spec, N):
    """
    Resolve node position specification to actual node index.
    
    Args:
        node_spec: Can be integer, "center", "center+offset", "center-offset", "N-offset"
        N: System size
    
    Returns:
        int: Actual node index
    """
    if isinstance(node_spec, int):
        return max(0, min(N-1, node_spec))
    
    if isinstance(node_spec, str):
        center = N // 2
        
        if node_spec == "center":
            return center
        elif node_spec.startswith("center+"):
            offset = int(node_spec[7:])
            return max(0, min(N-1, center + offset))
        elif node_spec.startswith("center-"):
            offset = int(node_spec[7:])
            return max(0, min(N-1, center - offset))
        elif node_spec.startswith("N-"):
            offset = int(node_spec[2:])
            return max(0, min(N-1, N - offset))
    
    raise ValueError(f"Invalid node specification: {node_spec}")

def load_survival_probability_data(devs, N, tesselation_func, survival_base_dir, noise_type, theta=None, survival_range="center_11"):
    """
    Load survival probability data directly from survival probability files.
    
    Args:
        devs: List of deviation values
        N: System size
        tesselation_func: Tessellation function
        survival_base_dir: Base directory for survival probability data
        noise_type: Type of noise
        theta: Theta parameter
        survival_range: Name of survival range to load
    
    Returns:
        List of survival probability arrays
    """
    print(f"\n[DATA] Loading survival probability data from '{survival_base_dir}'...")
    print(f"[DATA] Target survival range: {survival_range}")
    
    survival_probs = []
    
    for i, dev in enumerate(devs):
        print(f"\n[SURVIVAL {i+1}/{len(devs)}] Loading deviation: {dev}")
        
        # Get experiment directory for this deviation
        exp_dir, format_used = find_experiment_dir_flexible(
            tesselation_func, True, N, [dev], noise_type, survival_base_dir, theta
        )
        
        # Try to find survival probability files
        survival_files = [
            os.path.join(exp_dir, f"survival_{survival_range}.pkl"),
        ]
        
        # Try to load existing survival data
        loaded = False
        for survival_file in survival_files:
            try:
                if os.path.exists(survival_file):
                    with open(survival_file, 'rb') as f:
                        survival_values = pickle.load(f)
                    print(f"  [OK] Loaded {len(survival_values)} survival values from {os.path.relpath(survival_file, exp_dir)}")
                    survival_probs.append(survival_values)
                    loaded = True
                    break
            except Exception as e:
                print(f"  [WARNING] Could not load survival data from {survival_file}: {e}")
                continue
        
        # If not found directly, check inside samples subfolder (for compatibility)
        if not loaded:
            samples_folder = f"samples_{samples}"
            samples_dir = os.path.join(exp_dir, samples_folder)
            
            if os.path.exists(samples_dir):
                survival_files_samples = [
                    os.path.join(samples_dir, f"survival_{survival_range}.pkl"),
                ]
                
                for survival_file in survival_files_samples:
                    try:
                        if os.path.exists(survival_file):
                            with open(survival_file, 'rb') as f:
                                survival_values = pickle.load(f)
                            print(f"  [OK] Loaded {len(survival_values)} survival values from {os.path.relpath(survival_file, exp_dir)}")
                            survival_probs.append(survival_values)
                            loaded = True
                            break
                    except Exception as e:
                        print(f"  [WARNING] Could not load survival data from {survival_file}: {e}")
                        continue
        
        if not loaded:
            print(f"  [MISSING] Survival probability file not found in: {exp_dir}")
            print(f"  [MISSING] Also checked samples subfolder: {os.path.join(exp_dir, f'samples_{samples}')}")
            print(f"  [INFO] Expected file: survival_{survival_range}.pkl")
            survival_probs.append([])
    
    print(f"[OK] Survival probability data loading completed!")
    return survival_probs

def calculate_survival_probability_from_array(prob_array, from_node, to_node):
    """
    Calculate survival probability from a probability array.
    
    Args:
        prob_array: numpy array of probabilities
        from_node: Starting node index
        to_node: Ending node index
    
    Returns:
        float: Survival probability (sum of probabilities in range)
    """
    if from_node == to_node:
        return float(prob_array[from_node])
    else:
        return float(np.sum(prob_array[from_node:to_node+1]))

def calculate_survival_probabilities_for_range(mean_prob_data, survival_range, N):
    """
    Calculate survival probability for a specific range across all time steps.
    
    Args:
        mean_prob_data: List of mean probability arrays for each time step
        survival_range: Dictionary with range specification
        N: System size
    
    Returns:
        numpy array: survival probabilities for each time step
    """
    range_name = survival_range["name"]
    from_node_spec = survival_range["from_node"]
    to_node_spec = survival_range["to_node"]
    
    # Resolve node positions
    from_node = resolve_node_position(from_node_spec, N)
    to_node = resolve_node_position(to_node_spec, N)
    
    # Ensure from_node <= to_node
    if from_node > to_node:
        from_node, to_node = to_node, from_node
    
    print(f"    Range '{range_name}': nodes {from_node} to {to_node} ({to_node-from_node+1} nodes)")
    
    survival_probs = []
    for step, prob_array in enumerate(mean_prob_data):
        if prob_array is not None:
            survival_prob = calculate_survival_probability_from_array(prob_array, from_node, to_node)
            survival_probs.append(survival_prob)
        else:
            # Handle corrupted data
            survival_probs.append(None)
            print(f"      Step {step}: Corrupted data, survival probability set to None")
    
    return np.array(survival_probs)

def generate_survival_probability_data(devs, N, steps, tesselation_func, probdist_base_dir, survival_base_dir, noise_type, theta, survival_range="center_11"):
    """
    Generate survival probability data from existing probability distributions.
    
    Args:
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        tesselation_func: Tessellation function
        probdist_base_dir: Base directory for probability distributions
        survival_base_dir: Base directory for survival probability data
        noise_type: Type of noise
        theta: Theta parameter
        survival_range: Name of survival range to calculate
    
    Returns:
        List of survival probability arrays
    """
    if not SMART_LOADING_AVAILABLE:
        print("[ERROR] Cannot generate survival probabilities - smart loading functions not available")
        return []
    
    print(f"\n[GENERATE] Generating survival probability data for range '{survival_range}'...")
    print(f"[GENERATE] Loading full probability distributions from '{probdist_base_dir}'...")
    
    try:
        # Load full probability distributions using smart loading
        mean_results = load_mean_probability_distributions(
            tesselation_func, N, steps, devs, samples, probdist_base_dir, noise_type, theta
        )
        print(f"[OK] Loaded full probability distributions for {len(devs)} deviations")
    except Exception as e:
        print(f"[ERROR] Failed to load full probability distributions: {e}")
        return []
    
    # Find the survival range configuration
    survival_range_config = None
    for range_spec in SURVIVAL_RANGES:
        if range_spec["name"] == survival_range:
            survival_range_config = range_spec
            break
    
    if survival_range_config is None:
        print(f"[ERROR] Survival range '{survival_range}' not found in SURVIVAL_RANGES")
        return []
    
    # Calculate survival probabilities for each deviation
    survival_probs = []
    
    for dev_idx, dev in enumerate(devs):
        print(f"\n[GENERATE {dev_idx+1}/{len(devs)}] Calculating survival probabilities for deviation: {dev}")
        
        mean_prob_data = mean_results[dev_idx]
        if mean_prob_data is None or len(mean_prob_data) == 0:
            print(f"  [ERROR] No probability data available for deviation {dev}")
            survival_probs.append([])
            continue
        
        # Calculate survival probabilities for this deviation
        survival_values = calculate_survival_probabilities_for_range(
            mean_prob_data, survival_range_config, N
        )
        
        survival_probs.append(survival_values)
        
        # Save the generated survival probability data if enabled
        if SAVE_GENERATED_SURVIVAL_PROB:
            try:
                # Create target directory structure
                exp_dir, format_used = find_experiment_dir_flexible(
                    tesselation_func, True, N, [dev], noise_type, survival_base_dir, theta
                )
                
                os.makedirs(exp_dir, exist_ok=True)
                
                # Save survival probability data
                survival_file = os.path.join(exp_dir, f"survival_{survival_range}.pkl")
                with open(survival_file, 'wb') as f:
                    pickle.dump(survival_values, f)
                
                print(f"  [SAVED] Survival probabilities saved to: {os.path.relpath(survival_file, exp_dir)}")
                
            except Exception as e:
                print(f"  [WARNING] Failed to save survival probabilities for {dev}: {e}")
    
    print(f"[OK] Survival probability generation completed!")
    return survival_probs

def load_or_generate_survival_probability_data(devs, N, steps, tesselation_func, survival_base_dir, probdist_base_dir, noise_type, theta=None, survival_range="center_11"):
    """
    Load survival probability data, generating it if it doesn't exist and generation is enabled.
    
    Args:
        devs: List of deviation values
        N: System size
        steps: Number of time steps
        tesselation_func: Tessellation function
        survival_base_dir: Base directory for survival probability data
        probdist_base_dir: Base directory for probability distributions (for generation)
        noise_type: Type of noise
        theta: Theta parameter
        survival_range: Name of survival range to load/generate
    
    Returns:
        List of survival probability arrays
    """
    # First, try to load existing data
    survival_probs = load_survival_probability_data(devs, N, tesselation_func, survival_base_dir, noise_type, theta, survival_range)
    
    # Check if we have complete data
    missing_data_count = sum(1 for sp in survival_probs if len(sp) == 0)
    
    if missing_data_count == 0:
        print(f"[OK] All survival probability data found and loaded")
        return survival_probs
    
    if missing_data_count == len(devs):
        print(f"[INFO] No survival probability data found")
    else:
        print(f"[INFO] Partial survival probability data found ({len(devs) - missing_data_count}/{len(devs)} deviations)")
    
    # Generate missing data if enabled
    if GENERATE_SURVIVAL_PROB_IF_MISSING:
        print(f"[INFO] Generation enabled - generating missing survival probability data...")
        generated_probs = generate_survival_probability_data(
            devs, N, steps, tesselation_func, probdist_base_dir, survival_base_dir, noise_type, theta, survival_range
        )
        
        if len(generated_probs) == len(devs):
            print(f"[OK] Successfully generated survival probability data")
            return generated_probs
        else:
            print(f"[WARNING] Generation partially failed - using combination of loaded and generated data")
            # Combine loaded and generated data
            combined_probs = []
            for i, (loaded, generated) in enumerate(zip(survival_probs, generated_probs)):
                if len(loaded) > 0:
                    combined_probs.append(loaded)
                elif len(generated) > 0:
                    combined_probs.append(generated)
                else:
                    combined_probs.append([])
            return combined_probs
    else:
        print(f"[INFO] Generation disabled - using only loaded data")
        return survival_probs

# ============================================================================
# STANDARD DEVIATION LOADING FUNCTIONS
# ============================================================================

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_standard_deviation_vs_time(stds, devs, steps):
    """
    Plot standard deviation vs time for different noise deviations.
    """
    config = STD_PLOT_CONFIG
    if not ENABLE_PLOTTING or not config['enabled']:
        print("\n[PLOT] Standard deviation plotting disabled")
        return
    
    print("\n[PLOT] Creating standard deviation vs time plot...")
    try:
        if len(stds) > 0 and any(len(std) > 0 for std in stds):
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=config['figure_size'])
            
            for i, (std_values, dev) in enumerate(zip(stds, devs)):
                if len(std_values) > 0:
                    time_steps = list(range(len(std_values)))
                    
                    # Format dev for display using label config
                    if isinstance(dev, tuple) and len(dev) == 2:
                        if dev[1] <= 1.0 and dev[1] >= 0.0:
                            # New format: (max_dev, min_factor)
                            max_dev, min_factor = dev
                            dev_label = LABEL_CONFIG['new_format_template'].format(
                                max_dev=max_dev, min_val=max_dev*min_factor, 
                                prec=LABEL_CONFIG['precision_new_format'])
                        else:
                            # Legacy format: (min, max)
                            min_val, max_val = dev
                            dev_label = LABEL_CONFIG['legacy_format_template'].format(
                                min_val=min_val, max_val=max_val, 
                                prec=LABEL_CONFIG['precision_legacy_format'])
                    else:
                        # Single value format
                        dev_label = LABEL_CONFIG['single_value_template'].format(
                            dev=dev, prec=LABEL_CONFIG['precision_single_value'])
                    
                    plot_label = LABEL_CONFIG['plot_label_template'].format(dev_label=dev_label)
                    
                    # Handle zero values for log-log plot
                    if config['use_loglog']:
                        # Check if this is a zero standard deviation case (noiseless)
                        if all(s == 0 for s in std_values):
                            # For noiseless case (std = 0), plot at bottom of y-axis
                            filtered_times = [t for t in time_steps if t > 0]
                            filtered_stds = [config['epsilon_noiseless']] * len(filtered_times)
                            plt.loglog(filtered_times, filtered_stds, 
                                     label=f'{plot_label} (noiseless)', 
                                     marker=config['marker_square'], markersize=config['marker_square_size'], 
                                     linewidth=config['linewidth'], 
                                     linestyle=config['linestyle_noiseless'], alpha=config['alpha_noiseless'])
                        else:
                            # Remove zero values which can't be plotted on log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, std_values) if t > 0 and s > 0]
                            if filtered_data:
                                filtered_times, filtered_stds = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_stds, 
                                         label=plot_label, 
                                         marker=config['marker_circle'], markersize=config['markersize'], 
                                         linewidth=config['linewidth'], alpha=config['alpha_regular'])
                    else:
                        plt.plot(time_steps, std_values, 
                               label=plot_label, 
                               marker=config['marker_circle'], markersize=config['markersize'], 
                               linewidth=config['linewidth'], alpha=config['alpha_regular'])
            
            plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
            plt.ylabel(config['ylabel'], fontsize=config['fontsize_labels'])
            
            if config['use_loglog']:
                plt.title(config['title_loglog'], fontsize=config['fontsize_title'])
                plt.grid(True, alpha=config['grid_alpha'], which=config['grid_which'], 
                        linestyle=config['grid_linestyle'])
                plot_filename = config['filename_loglog'].format(N=N, steps=steps, samples=samples, theta=theta)
            else:
                plt.title(config['title_linear'], fontsize=config['fontsize_title'])
                plt.grid(True, alpha=config['grid_alpha'])
                plot_filename = config['filename_linear'].format(N=N, steps=steps, samples=samples, theta=theta)
            
            plt.legend(fontsize=config['fontsize_legend'])
            plt.tight_layout()
            
            # Save the plot (if enabled)
            if config['save_figure']:
                plt.savefig(plot_filename, dpi=config['dpi'], bbox_inches=config['bbox_inches'])
                print(f"[OK] Plot saved as '{plot_filename}'")
            
            # Show the plot
            plt.show()
            plot_type = "log-log" if config['use_loglog'] else "linear"
            saved_status = " and saved" if config['save_figure'] else ""
            print(f"[OK] Standard deviation plot displayed{saved_status}! (Scale: {plot_type})")
        else:
            print("[WARNING] No standard deviation data available for plotting")
    except Exception as e:
        print(f"[WARNING] Could not create plot: {e}")
        import traceback
        traceback.print_exc()

def plot_final_probability_distributions(final_results, devs, steps, N):
    """
    Plot final probability distributions for different noise deviations.
    final_results: List of final step probability distributions (one per deviation)
    """
    config = PROBDIST_PLOT_CONFIG
    if not ENABLE_PLOTTING or not config['enabled']:
        if not ENABLE_PLOTTING:
            print("\n[PLOT] Plotting disabled (ENABLE_PLOTTING=False)")
        else:
            print("\n[PLOT] Final probability distribution plotting disabled")
        return
    
    print("\n[PLOT] Creating final probability distribution plot...")
    try:
        if final_results and len(final_results) > 0:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=config['figure_size'])
            
            # Use the last time step (steps-1)
            final_step = steps - 1
            domain = np.arange(N) - N//2  # Center domain around 0
            
            for i, (final_prob_dist, dev) in enumerate(zip(final_results, devs)):
                if final_prob_dist is not None:
                    final_prob_dist = final_prob_dist.flatten()
                    
                    # Format dev for display using label config
                    if isinstance(dev, tuple) and len(dev) == 2:
                        if dev[1] <= 1.0 and dev[1] >= 0.0:
                            # New format: (max_dev, min_factor)
                            max_dev, min_factor = dev
                            dev_label = LABEL_CONFIG['new_format_template'].format(
                                max_dev=max_dev, min_val=max_dev*min_factor, 
                                prec=LABEL_CONFIG['precision_new_format'])
                        else:
                            # Legacy format: (min, max)
                            min_val, max_val = dev
                            dev_label = LABEL_CONFIG['legacy_format_template'].format(
                                min_val=min_val, max_val=max_val, 
                                prec=LABEL_CONFIG['precision_legacy_format'])
                    else:
                        # Single value format
                        dev_label = LABEL_CONFIG['single_value_template'].format(
                            dev=dev, prec=LABEL_CONFIG['precision_single_value'])
                    
                    plot_label = LABEL_CONFIG['plot_label_template'].format(dev_label=dev_label)
                    
                    # Plot the probability distribution with log y-axis
                    plt.semilogy(domain, final_prob_dist, 
                               label=plot_label, 
                               linewidth=config['linewidth'], alpha=config['alpha'])
            
            plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
            plt.ylabel(config['ylabel'], fontsize=config['fontsize_labels'])
            plt.title(config['title'].format(final_step=final_step), fontsize=config['fontsize_title'])
            
            # Set axis limits based on config
            if config['xlim']:
                plt.xlim(config['xlim'])
            if config['ylim_min'] is not None or config['ylim_max'] is not None:
                plt.ylim(config['ylim_min'], config['ylim_max'])
            
            plt.legend(fontsize=config['fontsize_legend'])
            plt.grid(True, alpha=config['grid_alpha'])
            plt.tight_layout()
            
            # Save the plot (if enabled)
            if config['save_figure']:
                plot_filename = config['filename'].format(N=N, steps=steps, samples=samples, theta=theta)
                plt.savefig(plot_filename, dpi=config['dpi'], bbox_inches=config['bbox_inches'])
                print(f"[OK] Probability distribution plot saved as '{plot_filename}'")
            
            # Show the plot
            plt.show()
            saved_status = " and saved" if config['save_figure'] else ""
            print(f"[OK] Final probability distribution plot displayed{saved_status}!")
        else:
            print("[WARNING] No mean probability distribution data available for plotting")
    except Exception as e:
        print(f"[WARNING] Could not create probability distribution plot: {e}")
        import traceback
        traceback.print_exc()

def plot_survival_probabilities(survival_probs, devs, steps, survival_range="center_11"):
    """
    Plot survival probabilities vs time for different noise deviations.
    
    Args:
        survival_probs: List of survival probability arrays (one per deviation)
        devs: List of deviation values
        steps: Number of time steps
        survival_range: Name of the survival range being plotted
    """
    config = SURVIVAL_PLOT_CONFIG
    if not ENABLE_PLOTTING or not config['enabled']:
        print("\n[PLOT] Survival probability plotting disabled")
        return
    
    print(f"\n[PLOT] Creating survival probability vs time plot for range '{survival_range}'...")
    try:
        if len(survival_probs) > 0 and any(len(sp) > 0 for sp in survival_probs):
            import matplotlib.pyplot as plt
            
            # Determine which plot type to create
            plot_types = []
            if not config['use_loglog'] and not config['use_semilogy']:
                plot_types.append('linear')
            if config['use_semilogy']:
                plot_types.append('semilogy')
            if config['use_loglog']:
                plot_types.append('loglog')
            
            for plot_type in plot_types:
                plt.figure(figsize=config['figure_size'])
                
                for i, (survival_values, dev) in enumerate(zip(survival_probs, devs)):
                    if len(survival_values) > 0:
                        time_steps = list(range(len(survival_values)))
                        
                        # Format dev for display using label config
                        if isinstance(dev, tuple) and len(dev) == 2:
                            if dev[1] <= 1.0 and dev[1] >= 0.0:
                                # New format: (max_dev, min_factor)
                                max_dev, min_factor = dev
                                dev_label = LABEL_CONFIG['new_format_template'].format(
                                    max_dev=max_dev, min_val=max_dev*min_factor, 
                                    prec=LABEL_CONFIG['precision_new_format'])
                            else:
                                # Legacy format: (min, max)
                                min_val, max_val = dev
                                dev_label = LABEL_CONFIG['legacy_format_template'].format(
                                    min_val=min_val, max_val=max_val, 
                                    prec=LABEL_CONFIG['precision_legacy_format'])
                        else:
                            # Single value format
                            dev_label = LABEL_CONFIG['single_value_template'].format(
                                dev=dev, prec=LABEL_CONFIG['precision_single_value'])
                        
                        plot_label = LABEL_CONFIG['plot_label_template'].format(dev_label=dev_label)
                        
                        # Plot based on type
                        if plot_type == 'linear':
                            # Filter out None values even for linear scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, survival_values) if s is not None]
                            if filtered_data:
                                filtered_times, filtered_survival = zip(*filtered_data)
                                plt.plot(filtered_times, filtered_survival, 
                                       label=plot_label, 
                                       linewidth=config['linewidth'], 
                                       alpha=config['alpha_regular'])
                        elif plot_type == 'semilogy':
                            # Filter out zero/negative/None values for log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, survival_values) if s is not None and s > 0]
                            if filtered_data:
                                filtered_times, filtered_survival = zip(*filtered_data)
                                plt.semilogy(filtered_times, filtered_survival, 
                                           label=plot_label, 
                                           linewidth=config['linewidth'], 
                                           alpha=config['alpha_regular'])
                        elif plot_type == 'loglog':
                            # Filter out zero/negative/None values for log scale
                            filtered_data = [(t, s) for t, s in zip(time_steps, survival_values) if t > 0 and s is not None and s > 0]
                            if filtered_data:
                                filtered_times, filtered_survival = zip(*filtered_data)
                                plt.loglog(filtered_times, filtered_survival, 
                                         label=plot_label, 
                                         linewidth=config['linewidth'], 
                                         alpha=config['alpha_regular'])
                
                # Set labels and title based on plot type
                if plot_type == 'linear':
                    plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
                    plt.ylabel(config['ylabel_linear'], fontsize=config['fontsize_labels'])
                    plt.title(config['title_linear'] + f" - {survival_range.replace('_', ' ').title()}", 
                             fontsize=config['fontsize_title'])
                    plot_filename = config['filename_linear'].format(N=N, steps=steps, samples=samples, theta=theta)
                elif plot_type == 'semilogy':
                    plt.xlabel(config['xlabel'], fontsize=config['fontsize_labels'])
                    plt.ylabel(config['ylabel_semilogy'], fontsize=config['fontsize_labels'])
                    plt.title(config['title_semilogy'] + f" - {survival_range.replace('_', ' ').title()}", 
                             fontsize=config['fontsize_title'])
                    plot_filename = config['filename_semilogy'].format(N=N, steps=steps, samples=samples, theta=theta)
                elif plot_type == 'loglog':
                    plt.xlabel(config['xlabel'] + " (log scale)", fontsize=config['fontsize_labels'])
                    plt.ylabel(config['ylabel_loglog'], fontsize=config['fontsize_labels'])
                    plt.title(config['title_loglog'] + f" - {survival_range.replace('_', ' ').title()}", 
                             fontsize=config['fontsize_title'])
                    plot_filename = config['filename_loglog'].format(N=N, steps=steps, samples=samples, theta=theta)
                    plt.grid(True, alpha=config['grid_alpha'], which=config['grid_which'], 
                            linestyle=config['grid_linestyle'])
                else:
                    plt.grid(True, alpha=config['grid_alpha'])
                
                if plot_type != 'loglog':
                    plt.grid(True, alpha=config['grid_alpha'])
                
                plt.legend(fontsize=config['fontsize_legend'])
                plt.tight_layout()
                
                # Save the plot (if enabled)
                if config['save_figure']:
                    # Insert survival range into filename
                    base_name, ext = os.path.splitext(plot_filename)
                    plot_filename_with_range = f"{base_name}_{survival_range}{ext}"
                    plt.savefig(plot_filename_with_range, dpi=config['dpi'], bbox_inches=config['bbox_inches'])
                    print(f"[OK] Survival probability plot saved as '{plot_filename_with_range}'")
                
                # Show the plot
                plt.show()
                
            saved_status = " and saved" if config['save_figure'] else ""
            scale_info = f" ({', '.join(plot_types)} scale{'s' if len(plot_types) > 1 else ''})"
            print(f"[OK] Survival probability plot{scale_info} displayed{saved_status}!")
        else:
            print("[WARNING] No survival probability data available for plotting")
    except Exception as e:
        print(f"[WARNING] Could not create survival probability plot: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that loads data and creates plots.
    """
    print("="*80)
    print("STATIC NOISE EXPERIMENT - PLOTTING ONLY")
    print("="*80)
    print(f"System size: N = {N}")
    print(f"Time steps: {steps}")
    print(f"Samples per deviation: {samples}")
    print(f"Deviation values: {devs}")
    print(f"Theta parameter: {theta}")
    print("")
    
    print("Configuration:")
    print(f"  ENABLE_PLOTTING = {ENABLE_PLOTTING}")
    print(f"  GENERATE_SURVIVAL_PROB_IF_MISSING = {GENERATE_SURVIVAL_PROB_IF_MISSING}")
    print(f"  SAVE_GENERATED_SURVIVAL_PROB = {SAVE_GENERATED_SURVIVAL_PROB}")
    print(f"  STD_PLOT enabled = {STD_PLOT_CONFIG['enabled']}")
    print(f"  STD_PLOT use_loglog = {STD_PLOT_CONFIG['use_loglog']}")
    print(f"  PROBDIST_PLOT enabled = {PROBDIST_PLOT_CONFIG['enabled']}")
    print(f"  SURVIVAL_PLOT enabled = {SURVIVAL_PLOT_CONFIG['enabled']}")
    print(f"  SURVIVAL_PLOT use_semilogy = {SURVIVAL_PLOT_CONFIG['use_semilogy']}")
    print(f"  SURVIVAL_PLOT use_loglog = {SURVIVAL_PLOT_CONFIG['use_loglog']}")
    print(f"  SURVIVAL_PLOT range = {SURVIVAL_PLOT_CONFIG['survival_range']}")
    print(f"  Save figures = {STD_PLOT_CONFIG['save_figure'] and PROBDIST_PLOT_CONFIG['save_figure'] and SURVIVAL_PLOT_CONFIG['save_figure']}")
    print("")
    
    # Step 1: Load standard deviation data directly
    print("[STEP 1] Loading standard deviation data...")
    try:
        stds = load_std_data_directly(devs, N, dummy_tesselation_func, std_base_dir, noise_type, theta)
        print(f"[OK] Standard deviation data ready for {len([s for s in stds if len(s) > 0])} / {len(devs)} deviations")
    except Exception as e:
        print(f"[ERROR] Failed to load standard deviation data: {e}")
        stds = []
    
    # Step 2: Load final step probability distributions (only if needed for plotting)
    final_results = []
    if PROBDIST_PLOT_CONFIG['enabled']:
        print("[STEP 2] Loading final step probability distributions...")
        try:
            final_results = load_final_step_probability_distributions(
                dummy_tesselation_func, N, steps, devs, probdist_base_dir, noise_type, theta
            )
            print(f"[OK] Loaded final step distributions for {len([r for r in final_results if r is not None])} / {len(devs)} deviations")
        except Exception as e:
            print(f"[ERROR] Failed to load final step probability distributions: {e}")
            final_results = []
    else:
        print("[STEP 2] Skipping final step probability distributions (PROBDIST_PLOT enabled=False)")
    
    # Step 3: Load or generate survival probability data (only if needed for plotting)
    survival_probs = []
    if SURVIVAL_PLOT_CONFIG['enabled']:
        print("[STEP 3] Loading or generating survival probability data...")
        try:
            survival_range = SURVIVAL_PLOT_CONFIG['survival_range']
            survival_probs = load_or_generate_survival_probability_data(
                devs, N, steps, dummy_tesselation_func, survival_base_dir, probdist_base_dir, noise_type, theta, survival_range
            )
            print(f"[OK] Prepared survival probabilities for {len([s for s in survival_probs if len(s) > 0])} / {len(devs)} deviations")
        except Exception as e:
            print(f"[ERROR] Failed to load/generate survival probability data: {e}")
            import traceback
            traceback.print_exc()
            survival_probs = []
    else:
        print("[STEP 3] Skipping survival probability data (SURVIVAL_PLOT enabled=False)")
    
    # Step 4: Create plots
    print("[STEP 4] Creating plots...")
    
    # Plot 1: Standard deviation vs time
    plot_standard_deviation_vs_time(stds, devs, steps)
    
    # Plot 2: Final probability distributions
    plot_final_probability_distributions(final_results, devs, steps, N)
    
    # Plot 3: Survival probabilities
    if SURVIVAL_PLOT_CONFIG['enabled'] and survival_probs:
        survival_range = SURVIVAL_PLOT_CONFIG['survival_range']
        plot_survival_probabilities(survival_probs, devs, steps, survival_range)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
