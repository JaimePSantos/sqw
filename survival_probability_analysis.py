#!/usr/bin/env python3

"""
Survival Probability Analysis for Quantum Walk Experiments

This script loads probability distributions from the cluster experiment results and calculates 
survival probabilities using the ProbabilityDistribution class implementation.

The survival probability is calculated as the sum of probabilities within a specified range 
[fromNode, toNode] at each time step for each deviation value.

Features:
- Loads existing mean probability distributions from experiments_data_samples_probDist
- Calculates survival probabilities for configurable node ranges
- Saves results to new folder experiments_data_samples_survivalProb
- Supports both single-node and range survival probability calculations
- Compatible with static noise experiment data structure
- Includes logging and progress tracking
- Memory-efficient processing with streaming data access
"""

import os
import pickle
import time
import numpy as np
import math
import logging
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Add QWAK to path for ProbabilityDistribution and State classes
sys.path.append(r'c:\Users\jaime\Documents\GitHub\QWAK\core')

try:
    from qwak.ProbabilityDistribution import ProbabilityDistribution
    from qwak.State import State
    print("[OK] Successfully imported QWAK ProbabilityDistribution and State classes")
except ImportError as e:
    print(f"[ERROR] Failed to import QWAK classes: {e}")
    print("Please ensure QWAK is properly installed and accessible")
    sys.exit(1)

# Import smart loading functions from the sqw project
try:
    from smart_loading_static import (
        load_mean_probability_distributions,
        check_mean_probability_distributions_exist,
        get_experiment_dir,
        find_experiment_dir_flexible
    )
    print("[OK] Successfully imported smart loading functions")
except ImportError as e:
    print(f"[ERROR] Failed to import smart loading functions: {e}")
    print("Please ensure you are running from the sqw project directory")
    sys.exit(1)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Survival probability calculation parameters
# Configure these ranges based on your analysis needs
SURVIVAL_RANGES = [
    {"name": "center_single", "from_node": "center", "to_node": "center"},  # Single center node
    {"name": "center_5", "from_node": "center-2", "to_node": "center+2"},   # 5 nodes around center
    {"name": "center_11", "from_node": "center-5", "to_node": "center+5"},  # 11 nodes around center
    {"name": "center_21", "from_node": "center-10", "to_node": "center+10"}, # 21 nodes around center
    {"name": "left_half", "from_node": 0, "to_node": "center"},             # Left half of system
    {"name": "right_half", "from_node": "center", "to_node": "N-1"},        # Right half of system
]

# Directory configuration
SOURCE_BASE_DIR = "experiments_data_samples_probDist"  # Source probability distributions
TARGET_BASE_DIR = "experiments_data_samples_survivalProb"  # Target survival probability data

# Experiment parameters (should match the original experiment)
N = 20000  # System size
steps = N//4  # Time steps
theta = math.pi/3  # Theta parameter for static noise

# Deviation values (should match the original experiment)
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.6),           # Medium noise range  
    (0, 0.8),           # Medium noise range  
    (0, 1),             # Medium noise range  
]

# Logging configuration
LOG_FILE = "survival_probability_analysis.log"
PROGRESS_UPDATE_INTERVAL = 100  # Update progress every N steps

# Plotting configuration
# Set ENABLE_PLOTTING = False to disable all plotting
# Set ENABLE_LOG_LINEAR_PLOTTING = True to create log-linear (semilogy) plots alongside linear plots
# Set ENABLE_LOG_LOG_PLOTTING = True to create log-log (loglog) plots alongside other plots
# Log-linear plots are useful for analyzing exponential decay behavior in survival probabilities
# Log-log plots are useful for analyzing power-law relationships and scaling behavior
ENABLE_PLOTTING = True  # Set to False to disable all plotting
ENABLE_LOG_LINEAR_PLOTTING = True  # Set to True to enable log-linear (semilogy) plots
ENABLE_LOG_LOG_PLOTTING = True  # Set to True to enable log-log (loglog) plots
PLOT_DPI = 300  # DPI for saved plots
PLOT_FORMAT = 'png'  # Format for saved plots (png, pdf, svg, etc.)
PLOT_FIGSIZE = (12, 8)  # Figure size for plots
COLORS = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Dummy tessellation function for static noise (tessellations are built-in)
def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging for the survival probability analysis"""
    
    # Create logger
    logger = logging.getLogger("survival_analysis")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(LOG_FILE, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# ============================================================================
# SURVIVAL PROBABILITY CALCULATION FUNCTIONS
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

def calculate_survival_probability_for_range(mean_prob_data, survival_range, N, logger):
    """
    Calculate survival probability for a specific range across all time steps.
    
    Args:
        mean_prob_data: List of mean probability arrays for each time step
        survival_range: Dictionary with range specification
        N: System size
        logger: Logger instance
    
    Returns:
        tuple: (range_name, survival_probabilities_array)
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
    
    logger.info(f"  Range '{range_name}': nodes {from_node} to {to_node} ({to_node-from_node+1} nodes)")
    
    survival_probs = []
    for step, prob_array in enumerate(mean_prob_data):
        if prob_array is not None:
            survival_prob = calculate_survival_probability_from_array(prob_array, from_node, to_node)
            survival_probs.append(survival_prob)
        else:
            # Handle corrupted data
            survival_probs.append(None)
            logger.warning(f"    Step {step}: Corrupted data, survival probability set to None")
    
    return range_name, np.array(survival_probs)

def process_deviation_survival_probabilities(dev, dev_idx, mean_prob_data, N, logger):
    """
    Process survival probabilities for all ranges for a single deviation.
    
    Args:
        dev: Deviation value
        dev_idx: Deviation index
        mean_prob_data: List of mean probability arrays for each time step
        N: System size
        logger: Logger instance
    
    Returns:
        dict: Dictionary with range names as keys and survival probability arrays as values
    """
    # Format dev_value for logging
    if isinstance(dev, (tuple, list)) and len(dev) == 2:
        dev_str = f"({dev[0]}, {dev[1]})"
    else:
        dev_str = str(dev)
    
    logger.info(f"Processing deviation {dev_idx+1}/{len(devs)}: {dev_str}")
    
    start_time = time.time()
    
    # Calculate survival probabilities for each range
    survival_results = {}
    
    for range_idx, survival_range in enumerate(SURVIVAL_RANGES):
        range_name, survival_probs = calculate_survival_probability_for_range(
            mean_prob_data, survival_range, N, logger
        )
        survival_results[range_name] = survival_probs
        
        # Log some statistics
        valid_probs = survival_probs[survival_probs != None]
        if len(valid_probs) > 0:
            mean_survival = np.mean(valid_probs)
            std_survival = np.std(valid_probs)
            logger.info(f"    {range_name}: mean={mean_survival:.6f}, std={std_survival:.6f}")
        else:
            logger.warning(f"    {range_name}: No valid data points")
    
    processing_time = time.time() - start_time
    logger.info(f"  Deviation {dev_idx+1} completed in {processing_time:.2f}s")
    
    return survival_results

# ============================================================================
# DATA SAVING FUNCTIONS
# ============================================================================

def save_survival_probability_data(dev, survival_results, target_dir, logger):
    """
    Save survival probability data for a single deviation.
    
    Args:
        dev: Deviation value
        survival_results: Dictionary with survival probability results
        target_dir: Target directory for saving
        logger: Logger instance
    """
    os.makedirs(target_dir, exist_ok=True)
    
    for range_name, survival_probs in survival_results.items():
        filename = f"survival_{range_name}.pkl"
        filepath = os.path.join(target_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(survival_probs, f)
            logger.info(f"    Saved {range_name} data to {filename}")
        except Exception as e:
            logger.error(f"    Failed to save {range_name} data: {e}")

def save_summary_data(target_base_dir, devs, survival_results_all, logger):
    """
    Save summary data including metadata and consolidated results.
    
    Args:
        target_base_dir: Base target directory
        devs: List of deviation values
        survival_results_all: Dictionary with all survival results
        logger: Logger instance
    """
    summary_dir = os.path.join(target_base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        "N": N,
        "steps": steps,
        "theta": theta,
        "devs": devs,
        "survival_ranges": SURVIVAL_RANGES,
        "creation_time": datetime.now().isoformat(),
        "source_dir": SOURCE_BASE_DIR,
    }
    
    metadata_file = os.path.join(summary_dir, "metadata.pkl")
    try:
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_file}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
    
    # Save consolidated results for easy loading
    consolidated_file = os.path.join(summary_dir, "all_survival_probabilities.pkl")
    try:
        with open(consolidated_file, 'wb') as f:
            pickle.dump(survival_results_all, f)
        logger.info(f"Saved consolidated results to {consolidated_file}")
    except Exception as e:
        logger.error(f"Failed to save consolidated results: {e}")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def create_survival_probability_plots(survival_results_all, devs, target_base_dir, logger):
    """
    Create plots for survival probability analysis.
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        target_base_dir: Base target directory for saving plots
        logger: Logger instance
    """
    if not ENABLE_PLOTTING:
        logger.info("Plotting disabled - skipping plot generation")
        return
    
    logger.info("Creating survival probability plots...")
    
    # Create plots directory
    plots_dir = os.path.join(target_base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get time steps array
    time_steps = np.arange(steps)
    
    # Create plots for each survival range
    for range_spec in SURVIVAL_RANGES:
        range_name = range_spec["name"]
        
        # Create both linear and log-linear plots for each range
        create_range_plots(survival_results_all, devs, range_name, time_steps, plots_dir, logger)
    
    # Create comparison plots (all ranges for each deviation)
    create_comparison_plots(survival_results_all, devs, time_steps, plots_dir, logger)
    
    logger.info(f"Plots saved to: {plots_dir}")

def create_range_plots(survival_results_all, devs, range_name, time_steps, plots_dir, logger):
    """
    Create plots for a specific survival range across all deviations.
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        range_name: Name of the survival range
        time_steps: Array of time steps
        plots_dir: Directory to save plots
        logger: Logger instance
    """
    # Linear scale plot
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    
    for dev_idx, (dev, color) in enumerate(zip(devs, COLORS)):
        if dev_idx >= len(survival_results_all):
            continue
            
        survival_data = survival_results_all[dev_idx]["survival_data"]
        if range_name not in survival_data:
            continue
            
        survival_probs = survival_data[range_name]
        
        # Filter out None values
        valid_indices = survival_probs != None
        valid_times = time_steps[valid_indices]
        valid_probs = survival_probs[valid_indices]
        
        # Format deviation label
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            dev_label = f"σ ∈ [{dev[0]}, {dev[1]}]"
        else:
            dev_label = f"σ = {dev}"
        
        ax.plot(valid_times, valid_probs, color=color, linewidth=2, 
               label=dev_label, alpha=0.8)
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Probability vs Time - {range_name.replace("_", " ").title()}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save linear plot
    linear_filename = f"survival_{range_name}_linear.{PLOT_FORMAT}"
    linear_filepath = os.path.join(plots_dir, linear_filename)
    plt.tight_layout()
    plt.savefig(linear_filepath, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Log-linear scale plot (if enabled)
    if ENABLE_LOG_LINEAR_PLOTTING:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        for dev_idx, (dev, color) in enumerate(zip(devs, COLORS)):
            if dev_idx >= len(survival_results_all):
                continue
                
            survival_data = survival_results_all[dev_idx]["survival_data"]
            if range_name not in survival_data:
                continue
                
            survival_probs = survival_data[range_name]
            
            # Filter out None values and zero/negative values for log scale
            valid_indices = (survival_probs != None) & (survival_probs > 0)
            valid_times = time_steps[valid_indices]
            valid_probs = survival_probs[valid_indices]
            
            if len(valid_probs) == 0:
                logger.warning(f"No positive values for {range_name}, dev {dev} - skipping log plot")
                continue
            
            # Format deviation label
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_label = f"σ ∈ [{dev[0]}, {dev[1]}]"
            else:
                dev_label = f"σ = {dev}"
            
            ax.semilogy(valid_times, valid_probs, color=color, linewidth=2, 
                       label=dev_label, alpha=0.8)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
        ax.set_title(f'Survival Probability vs Time (Log Scale) - {range_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save log-linear plot
        log_filename = f"survival_{range_name}_loglinear.{PLOT_FORMAT}"
        log_filepath = os.path.join(plots_dir, log_filename)
        plt.tight_layout()
        plt.savefig(log_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Created plots for {range_name}: {linear_filename}, {log_filename}")
    else:
        logger.info(f"  Created plot for {range_name}: {linear_filename}")
    
    # Log-log scale plot (if enabled)
    if ENABLE_LOG_LOG_PLOTTING:
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        for dev_idx, (dev, color) in enumerate(zip(devs, COLORS)):
            if dev_idx >= len(survival_results_all):
                continue
                
            survival_data = survival_results_all[dev_idx]["survival_data"]
            if range_name not in survival_data:
                continue
                
            survival_probs = survival_data[range_name]
            
            # Filter out None values, zero/negative values, and zero time steps for log-log scale
            valid_indices = (survival_probs != None) & (survival_probs > 0) & (time_steps > 0)
            valid_times = time_steps[valid_indices]
            valid_probs = survival_probs[valid_indices]
            
            if len(valid_probs) == 0:
                logger.warning(f"No positive values for {range_name}, dev {dev} - skipping log-log plot")
                continue
            
            # Format deviation label
            if isinstance(dev, (tuple, list)) and len(dev) == 2:
                dev_label = f"σ ∈ [{dev[0]}, {dev[1]}]"
            else:
                dev_label = f"σ = {dev}"
            
            ax.loglog(valid_times, valid_probs, color=color, linewidth=2, 
                     label=dev_label, alpha=0.8)
        
        ax.set_xlabel('Time Steps (log scale)', fontsize=12)
        ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
        ax.set_title(f'Survival Probability vs Time (Log-Log Scale) - {range_name.replace("_", " ").title()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save log-log plot
        loglog_filename = f"survival_{range_name}_loglog.{PLOT_FORMAT}"
        loglog_filepath = os.path.join(plots_dir, loglog_filename)
        plt.tight_layout()
        plt.savefig(loglog_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        if ENABLE_LOG_LINEAR_PLOTTING:
            logger.info(f"  Created plots for {range_name}: {linear_filename}, {log_filename}, {loglog_filename}")
        else:
            logger.info(f"  Created plots for {range_name}: {linear_filename}, {loglog_filename}")
    elif ENABLE_LOG_LINEAR_PLOTTING:
        logger.info(f"  Created plots for {range_name}: {linear_filename}, {log_filename}")
    else:
        logger.info(f"  Created plot for {range_name}: {linear_filename}")

def create_comparison_plots(survival_results_all, devs, time_steps, plots_dir, logger):
    """
    Create comparison plots showing all ranges for each deviation.
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        time_steps: Array of time steps
        plots_dir: Directory to save plots
        logger: Logger instance
    """
    for dev_idx, dev in enumerate(devs):
        if dev_idx >= len(survival_results_all):
            continue
            
        survival_data = survival_results_all[dev_idx]["survival_data"]
        
        # Format deviation for filename and title
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            dev_str = f"dev_min{dev[0]:.3f}_max{dev[1]:.3f}"
            dev_title = f"σ ∈ [{dev[0]}, {dev[1]}]"
        else:
            dev_str = f"dev_{dev:.3f}"
            dev_title = f"σ = {dev}"
        
        # Linear scale comparison plot
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        for range_idx, range_spec in enumerate(SURVIVAL_RANGES):
            range_name = range_spec["name"]
            if range_name not in survival_data:
                continue
                
            survival_probs = survival_data[range_name]
            
            # Filter out None values
            valid_indices = survival_probs != None
            valid_times = time_steps[valid_indices]
            valid_probs = survival_probs[valid_indices]
            
            color = COLORS[range_idx % len(COLORS)]
            ax.plot(valid_times, valid_probs, color=color, linewidth=2, 
                   label=range_name.replace("_", " ").title(), alpha=0.8)
        
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.set_title(f'Survival Probability Comparison - {dev_title}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save linear comparison plot
        linear_filename = f"survival_comparison_{dev_str}_linear.{PLOT_FORMAT}"
        linear_filepath = os.path.join(plots_dir, linear_filename)
        plt.tight_layout()
        plt.savefig(linear_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        # Log-linear scale comparison plot (if enabled)
        if ENABLE_LOG_LINEAR_PLOTTING:
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            
            for range_idx, range_spec in enumerate(SURVIVAL_RANGES):
                range_name = range_spec["name"]
                if range_name not in survival_data:
                    continue
                    
                survival_probs = survival_data[range_name]
                
                # Filter out None values and zero/negative values for log scale
                valid_indices = (survival_probs != None) & (survival_probs > 0)
                valid_times = time_steps[valid_indices]
                valid_probs = survival_probs[valid_indices]
                
                if len(valid_probs) == 0:
                    continue
                
                color = COLORS[range_idx % len(COLORS)]
                ax.semilogy(valid_times, valid_probs, color=color, linewidth=2, 
                           label=range_name.replace("_", " ").title(), alpha=0.8)
            
            ax.set_xlabel('Time Steps', fontsize=12)
            ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
            ax.set_title(f'Survival Probability Comparison (Log Scale) - {dev_title}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save log-linear comparison plot
            log_filename = f"survival_comparison_{dev_str}_loglinear.{PLOT_FORMAT}"
            log_filepath = os.path.join(plots_dir, log_filename)
            plt.tight_layout()
            plt.savefig(log_filepath, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Created comparison plots for {dev_title}: {linear_filename}, {log_filename}")
        else:
            logger.info(f"  Created comparison plot for {dev_title}: {linear_filename}")
        
        # Log-log scale comparison plot (if enabled)
        if ENABLE_LOG_LOG_PLOTTING:
            fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
            
            for range_idx, range_spec in enumerate(SURVIVAL_RANGES):
                range_name = range_spec["name"]
                if range_name not in survival_data:
                    continue
                    
                survival_probs = survival_data[range_name]
                
                # Filter out None values, zero/negative values, and zero time steps for log-log scale
                valid_indices = (survival_probs != None) & (survival_probs > 0) & (time_steps > 0)
                valid_times = time_steps[valid_indices]
                valid_probs = survival_probs[valid_indices]
                
                if len(valid_probs) == 0:
                    continue
                
                color = COLORS[range_idx % len(COLORS)]
                ax.loglog(valid_times, valid_probs, color=color, linewidth=2, 
                         label=range_name.replace("_", " ").title(), alpha=0.8)
            
            ax.set_xlabel('Time Steps (log scale)', fontsize=12)
            ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
            ax.set_title(f'Survival Probability Comparison (Log-Log Scale) - {dev_title}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save log-log comparison plot
            loglog_filename = f"survival_comparison_{dev_str}_loglog.{PLOT_FORMAT}"
            loglog_filepath = os.path.join(plots_dir, loglog_filename)
            plt.tight_layout()
            plt.savefig(loglog_filepath, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            if ENABLE_LOG_LINEAR_PLOTTING:
                logger.info(f"  Created comparison plots for {dev_title}: {linear_filename}, {log_filename}, {loglog_filename}")
            else:
                logger.info(f"  Created comparison plots for {dev_title}: {linear_filename}, {loglog_filename}")
        elif ENABLE_LOG_LINEAR_PLOTTING:
            logger.info(f"  Created comparison plots for {dev_title}: {linear_filename}, {log_filename}")
        else:
            logger.info(f"  Created comparison plot for {dev_title}: {linear_filename}")

def create_decay_analysis_plots(survival_results_all, devs, target_base_dir, logger):
    """
    Create specialized plots for analyzing decay rates (useful for log-linear analysis).
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        target_base_dir: Base target directory for saving plots
        logger: Logger instance
    """
    if not ENABLE_PLOTTING or not ENABLE_LOG_LINEAR_PLOTTING:
        return
    
    logger.info("Creating decay analysis plots...")
    
    # Create plots directory
    plots_dir = os.path.join(target_base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get time steps array
    time_steps = np.arange(steps)
    
    # Create decay rate heatmap for each range
    for range_spec in SURVIVAL_RANGES:
        range_name = range_spec["name"]
        
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        decay_rates = []
        dev_labels = []
        
        for dev_idx, dev in enumerate(devs):
            if dev_idx >= len(survival_results_all):
                continue
                
            survival_data = survival_results_all[dev_idx]["survival_data"]
            if range_name not in survival_data:
                continue
                
            survival_probs = survival_data[range_name]
            
            # Calculate decay rate (negative slope in log space)
            valid_indices = (survival_probs != None) & (survival_probs > 0)
            valid_times = time_steps[valid_indices]
            valid_probs = survival_probs[valid_indices]
            
            if len(valid_probs) > 10:  # Need sufficient points for fit
                # Fit linear model in log space
                log_probs = np.log(valid_probs)
                coeffs = np.polyfit(valid_times, log_probs, 1)
                decay_rate = -coeffs[0]  # Negative slope gives decay rate
                decay_rates.append(decay_rate)
                
                # Format deviation label
                if isinstance(dev, (tuple, list)) and len(dev) == 2:
                    dev_label = f"[{dev[0]}, {dev[1]}]"
                else:
                    dev_label = f"{dev}"
                dev_labels.append(dev_label)
        
        if decay_rates:
            # Create bar plot of decay rates
            bars = ax.bar(range(len(decay_rates)), decay_rates, color=COLORS[:len(decay_rates)])
            ax.set_xlabel('Deviation', fontsize=12)
            ax.set_ylabel('Decay Rate', fontsize=12)
            ax.set_title(f'Survival Probability Decay Rates - {range_name.replace("_", " ").title()}', fontsize=14)
            ax.set_xticks(range(len(dev_labels)))
            ax.set_xticklabels(dev_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, decay_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Save decay analysis plot
            decay_filename = f"survival_decay_rates_{range_name}.{PLOT_FORMAT}"
            decay_filepath = os.path.join(plots_dir, decay_filename)
            plt.tight_layout()
            plt.savefig(decay_filepath, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Created decay analysis plot: {decay_filename}")
        else:
            plt.close()
            logger.warning(f"  No valid data for decay analysis of {range_name}")

def create_power_law_analysis_plots(survival_results_all, devs, target_base_dir, logger):
    """
    Create specialized plots for analyzing power-law scaling (useful for log-log analysis).
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        target_base_dir: Base target directory for saving plots
        logger: Logger instance
    """
    if not ENABLE_PLOTTING or not ENABLE_LOG_LOG_PLOTTING:
        return
    
    logger.info("Creating power-law analysis plots...")
    
    # Create plots directory
    plots_dir = os.path.join(target_base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get time steps array
    time_steps = np.arange(steps)
    
    # Create power-law exponent analysis for each range
    for range_spec in SURVIVAL_RANGES:
        range_name = range_spec["name"]
        
        fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
        
        power_law_exponents = []
        dev_labels = []
        
        for dev_idx, dev in enumerate(devs):
            if dev_idx >= len(survival_results_all):
                continue
                
            survival_data = survival_results_all[dev_idx]["survival_data"]
            if range_name not in survival_data:
                continue
                
            survival_probs = survival_data[range_name]
            
            # Calculate power-law exponent (slope in log-log space)
            valid_indices = (survival_probs != None) & (survival_probs > 0) & (time_steps > 0)
            valid_times = time_steps[valid_indices]
            valid_probs = survival_probs[valid_indices]
            
            if len(valid_probs) > 10:  # Need sufficient points for fit
                # Fit linear model in log-log space
                log_times = np.log(valid_times)
                log_probs = np.log(valid_probs)
                coeffs = np.polyfit(log_times, log_probs, 1)
                power_law_exponent = coeffs[0]  # Slope gives power-law exponent
                power_law_exponents.append(power_law_exponent)
                
                # Format deviation label
                if isinstance(dev, (tuple, list)) and len(dev) == 2:
                    dev_label = f"[{dev[0]}, {dev[1]}]"
                else:
                    dev_label = f"{dev}"
                dev_labels.append(dev_label)
        
        if power_law_exponents:
            # Create bar plot of power-law exponents
            bars = ax.bar(range(len(power_law_exponents)), power_law_exponents, color=COLORS[:len(power_law_exponents)])
            ax.set_xlabel('Deviation', fontsize=12)
            ax.set_ylabel('Power-Law Exponent', fontsize=12)
            ax.set_title(f'Survival Probability Power-Law Exponents - {range_name.replace("_", " ").title()}', fontsize=14)
            ax.set_xticks(range(len(dev_labels)))
            ax.set_xticklabels(dev_labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels on bars
            for bar, exponent in zip(bars, power_law_exponents):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{exponent:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
            
            # Save power-law analysis plot
            powerlaw_filename = f"survival_powerlaw_exponents_{range_name}.{PLOT_FORMAT}"
            powerlaw_filepath = os.path.join(plots_dir, powerlaw_filename)
            plt.tight_layout()
            plt.savefig(powerlaw_filepath, dpi=PLOT_DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Created power-law analysis plot: {powerlaw_filename}")
        else:
            plt.close()
            logger.warning(f"  No valid data for power-law analysis of {range_name}")

def create_scaling_comparison_plots(survival_results_all, devs, target_base_dir, logger):
    """
    Create plots comparing different scaling behaviors (linear, log-linear, log-log).
    
    Args:
        survival_results_all: Dictionary with all survival results
        devs: List of deviation values
        target_base_dir: Base target directory for saving plots
        logger: Logger instance
    """
    if not ENABLE_PLOTTING:
        return
    
    logger.info("Creating scaling comparison plots...")
    
    # Create plots directory
    plots_dir = os.path.join(target_base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get time steps array
    time_steps = np.arange(steps)
    
    # Create scaling comparison for each deviation
    for dev_idx, dev in enumerate(devs):
        if dev_idx >= len(survival_results_all):
            continue
            
        survival_data = survival_results_all[dev_idx]["survival_data"]
        
        # Format deviation for filename and title
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            dev_str = f"dev_min{dev[0]:.3f}_max{dev[1]:.3f}"
            dev_title = f"σ ∈ [{dev[0]}, {dev[1]}]"
        else:
            dev_str = f"dev_{dev:.3f}"
            dev_title = f"σ = {dev}"
        
        # Create subplot figure with different scales
        scales_to_plot = ['linear']
        if ENABLE_LOG_LINEAR_PLOTTING:
            scales_to_plot.append('log-linear')
        if ENABLE_LOG_LOG_PLOTTING:
            scales_to_plot.append('log-log')
        
        n_plots = len(scales_to_plot)
        fig, axes = plt.subplots(1, n_plots, figsize=(PLOT_FIGSIZE[0] * n_plots, PLOT_FIGSIZE[1]))
        
        if n_plots == 1:
            axes = [axes]
        
        # Select a representative range for scaling comparison
        range_name = "center_11"  # Use center_11 as representative
        if range_name not in survival_data:
            range_name = list(survival_data.keys())[0]  # Fallback to first available range
        
        survival_probs = survival_data[range_name]
        
        for i, (scale, ax) in enumerate(zip(scales_to_plot, axes)):
            if scale == 'linear':
                valid_indices = survival_probs != None
                valid_times = time_steps[valid_indices]
                valid_probs = survival_probs[valid_indices]
                
                ax.plot(valid_times, valid_probs, color='blue', linewidth=2, alpha=0.8)
                ax.set_ylabel('Survival Probability', fontsize=12)
                
            elif scale == 'log-linear':
                valid_indices = (survival_probs != None) & (survival_probs > 0)
                valid_times = time_steps[valid_indices]
                valid_probs = survival_probs[valid_indices]
                
                if len(valid_probs) > 0:
                    ax.semilogy(valid_times, valid_probs, color='red', linewidth=2, alpha=0.8)
                ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
                
            elif scale == 'log-log':
                valid_indices = (survival_probs != None) & (survival_probs > 0) & (time_steps > 0)
                valid_times = time_steps[valid_indices]
                valid_probs = survival_probs[valid_indices]
                
                if len(valid_probs) > 0:
                    ax.loglog(valid_times, valid_probs, color='green', linewidth=2, alpha=0.8)
                ax.set_ylabel('Survival Probability (log scale)', fontsize=12)
                ax.set_xlabel('Time Steps (log scale)', fontsize=12)
            
            if scale != 'log-log':
                ax.set_xlabel('Time Steps', fontsize=12)
                
            ax.set_title(f'{scale.title()} Scale', fontsize=12)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Scaling Comparison - {range_name.replace("_", " ").title()} Range\\nNoise: {dev_title}', fontsize=14)
        plt.tight_layout()
        
        # Save scaling comparison plot
        scaling_filename = f"survival_scaling_comparison_{dev_str}.{PLOT_FORMAT}"
        scaling_filepath = os.path.join(plots_dir, scaling_filename)
        plt.savefig(scaling_filepath, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Created scaling comparison plot: {scaling_filename}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_survival_probability_analysis():
    """
    Main function to run survival probability analysis on existing probability distributions.
    """
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("SURVIVAL PROBABILITY ANALYSIS STARTED")
    logger.info("=" * 80)
    logger.info(f"System parameters: N={N}, steps={steps}, theta={theta}")
    logger.info(f"Source directory: {SOURCE_BASE_DIR}")
    logger.info(f"Target directory: {TARGET_BASE_DIR}")
    logger.info(f"Deviation values: {devs}")
    logger.info(f"Survival ranges: {len(SURVIVAL_RANGES)} ranges configured")
    
    for i, range_spec in enumerate(SURVIVAL_RANGES):
        logger.info(f"  Range {i+1}: {range_spec['name']} = [{range_spec['from_node']}, {range_spec['to_node']}]")
    
    start_time = time.time()
    
    # Check if source data exists
    logger.info("\n" + "=" * 40)
    logger.info("CHECKING SOURCE DATA AVAILABILITY")
    logger.info("=" * 40)
    
    if not check_mean_probability_distributions_exist(
        dummy_tesselation_func, N, steps, devs, 
        SOURCE_BASE_DIR, "static_noise", theta
    ):
        logger.error("Source probability distributions not found!")
        logger.error("Please run the cluster experiment first to generate probability distributions.")
        return False
    
    logger.info("Source data verified - proceeding with analysis")
    
    # Load probability distributions
    logger.info("\n" + "=" * 40)
    logger.info("LOADING PROBABILITY DISTRIBUTIONS")
    logger.info("=" * 40)
    
    try:
        mean_results = load_mean_probability_distributions(
            dummy_tesselation_func, N, steps, devs,
            SOURCE_BASE_DIR, "static_noise", theta
        )
        logger.info(f"Successfully loaded probability distributions for {len(devs)} deviations")
    except Exception as e:
        logger.error(f"Failed to load probability distributions: {e}")
        return False
    
    # Process survival probabilities for each deviation
    logger.info("\n" + "=" * 40)
    logger.info("CALCULATING SURVIVAL PROBABILITIES")
    logger.info("=" * 40)
    
    survival_results_all = {}
    
    for dev_idx, dev in enumerate(devs):
        mean_prob_data = mean_results[dev_idx]
        
        # Calculate survival probabilities for this deviation
        survival_results = process_deviation_survival_probabilities(
            dev, dev_idx, mean_prob_data, N, logger
        )
        
        # Store results
        survival_results_all[dev_idx] = {
            "dev_value": dev,
            "survival_data": survival_results
        }
        
        # Save data for this deviation
        logger.info(f"  Saving data for deviation {dev_idx+1}...")
        
        # Format deviation for directory name
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            dev_str = f"dev_min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"dev_{dev:.3f}"
        
        # Create target directory structure similar to source
        target_dev_dir = os.path.join(
            TARGET_BASE_DIR, 
            "dummy_tesselation_func_static_noise",
            f"theta_{theta:.6f}",
            dev_str,
            f"N_{N}"
        )
        
        save_survival_probability_data(dev, survival_results, target_dev_dir, logger)
    
    # Save summary data
    logger.info("\n" + "=" * 40)
    logger.info("SAVING SUMMARY DATA")
    logger.info("=" * 40)
    
    save_summary_data(TARGET_BASE_DIR, devs, survival_results_all, logger)
    
    # Create plots
    logger.info("\n" + "=" * 40)
    logger.info("CREATING PLOTS")
    logger.info("=" * 40)
    
    try:
        create_survival_probability_plots(survival_results_all, devs, TARGET_BASE_DIR, logger)
        create_decay_analysis_plots(survival_results_all, devs, TARGET_BASE_DIR, logger)
        create_power_law_analysis_plots(survival_results_all, devs, TARGET_BASE_DIR, logger)
        create_scaling_comparison_plots(survival_results_all, devs, TARGET_BASE_DIR, logger)
        logger.info("Plot generation completed successfully")
    except Exception as e:
        logger.error(f"Failed to create plots: {e}")
        logger.warning("Continuing without plots...")
    
    # Analysis complete
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("SURVIVAL PROBABILITY ANALYSIS COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Processed {len(devs)} deviations")
    logger.info(f"Calculated {len(SURVIVAL_RANGES)} survival ranges per deviation")
    logger.info(f"Results saved to: {TARGET_BASE_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    
    print("\n=== SURVIVAL PROBABILITY ANALYSIS SUMMARY ===")
    print(f"✓ Successfully processed {len(devs)} deviations")
    print(f"✓ Calculated {len(SURVIVAL_RANGES)} survival probability ranges")
    print(f"✓ Results saved to: {TARGET_BASE_DIR}")
    if ENABLE_PLOTTING:
        print(f"✓ Plots saved to: {os.path.join(TARGET_BASE_DIR, 'plots')}")
        plot_types = []
        if True:  # Always have linear
            plot_types.append("linear")
        if ENABLE_LOG_LINEAR_PLOTTING:
            plot_types.append("log-linear")
        if ENABLE_LOG_LOG_PLOTTING:
            plot_types.append("log-log")
        print(f"✓ Generated {', '.join(plot_types)} plots")
        
        if ENABLE_LOG_LINEAR_PLOTTING or ENABLE_LOG_LOG_PLOTTING:
            analysis_types = []
            if ENABLE_LOG_LINEAR_PLOTTING:
                analysis_types.append("decay rate analysis")
            if ENABLE_LOG_LOG_PLOTTING:
                analysis_types.append("power-law analysis")
            print(f"✓ Included {', '.join(analysis_types)}")
    else:
        print("ⓘ Plotting disabled")
    print(f"✓ Total time: {total_time:.2f} seconds")
    print(f"✓ Log file: {LOG_FILE}")
    
    print("\n=== AVAILABLE SURVIVAL RANGES ===")
    for i, range_spec in enumerate(SURVIVAL_RANGES):
        from_resolved = resolve_node_position(range_spec["from_node"], N)
        to_resolved = resolve_node_position(range_spec["to_node"], N)
        print(f"{i+1}. {range_spec['name']}: nodes {from_resolved}-{to_resolved} ({to_resolved-from_resolved+1} nodes)")
    
    print("\n=== DATA STRUCTURE ===")
    print("Individual files: [target_dir]/[dev_folder]/survival_[range_name].pkl")
    print("Summary data: [target_dir]/summary/all_survival_probabilities.pkl")
    print("Metadata: [target_dir]/summary/metadata.pkl")
    if ENABLE_PLOTTING:
        print("Plots: [target_dir]/plots/")
        print("  - survival_[range]_linear.png: Linear scale plots")
        if ENABLE_LOG_LINEAR_PLOTTING:
            print("  - survival_[range]_loglinear.png: Log-linear scale plots")
            print("  - survival_comparison_[dev]_linear.png: Range comparison (linear)")
            print("  - survival_comparison_[dev]_loglinear.png: Range comparison (log-linear)")
            print("  - survival_decay_rates_[range].png: Decay rate analysis")
        if ENABLE_LOG_LOG_PLOTTING:
            print("  - survival_[range]_loglog.png: Log-log scale plots")
            print("  - survival_comparison_[dev]_loglog.png: Range comparison (log-log)")
            print("  - survival_powerlaw_exponents_[range].png: Power-law exponent analysis")
            print("  - survival_scaling_comparison_[dev].png: Multi-scale comparison plots")
    
    return True

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        success = run_survival_probability_analysis()
        if success:
            print("\n🎉 Survival probability analysis completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Survival probability analysis failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
