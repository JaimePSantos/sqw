"""
Multiprocessing worker functions for static noise quantum walk experiments.

This module contains the worker functions that run in separate processes
for sample computation and mean probability calculation.
"""

import os
import sys
import time
import gc
import traceback
import pickle
import numpy as np
from typing import Dict, Any, Tuple

from .experiment_logging import setup_process_logging
from .system_monitor import log_system_resources, log_resource_usage


def dummy_tesselation_func(N):
    """Dummy tessellation function for static noise (tessellations are built-in)"""
    return None


def compute_dev_samples(dev_args: Tuple) -> Dict[str, Any]:
    """Worker function to compute samples for a single deviation value in a separate process"""
    dev, process_id, N, steps, samples, theta, initial_state_kwargs = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(dev_str, process_id)
    
    try:
        logger.info(f"Starting computation for deviation {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}, theta={theta:.4f}")
        
        # Import required modules (each process needs its own imports)
        from sqw.experiments_sparse import running_streaming_sparse
        from smart_loading_static import get_experiment_dir
        
        # Setup experiment directory - handle new deviation format
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # Direct (minVal, maxVal) format
            min_val, max_val = dev
            has_noise = max_val > 0
        else:
            # Single value format
            has_noise = dev > 0
        
        # With unified structure, we always include noise_params (including 0 for no noise)
        noise_params = [dev]
        exp_dir = get_experiment_dir(dummy_tesselation_func, has_noise, N, 
                                   noise_params=noise_params, noise_type="static_noise", 
                                   base_dir="experiments_data_samples", theta=theta)
        os.makedirs(exp_dir, exist_ok=True)
        
        logger.info(f"Experiment directory: {exp_dir}")
        
        dev_start_time = time.time()
        dev_computed_samples = 0
        
        for sample_idx in range(samples):
            sample_start_time = time.time()
            
            # Check if this sample already exists (all step files)
            sample_exists = True
            for step_idx in range(steps):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                sample_file = os.path.join(step_dir, f"sample_{sample_idx}.pkl")
                if not os.path.exists(sample_file):
                    sample_exists = False
                    break
            
            if sample_exists:
                logger.info(f"Sample {sample_idx+1}/{samples} already exists, skipping...")
                dev_computed_samples += 1
                continue
            
            logger.info(f"Computing sample {sample_idx+1}/{samples}...")
            
            # Extract initial nodes from initial_state_kwargs
            initial_nodes = initial_state_kwargs.get('nodes', [])
            
            # Create step callback function for streaming saves
            def save_step_callback(step_idx, state):
                step_dir = os.path.join(exp_dir, f"step_{step_idx}")
                os.makedirs(step_dir, exist_ok=True)
                sample_file = os.path.join(step_dir, f"sample_{sample_idx}.pkl")
                
                with open(sample_file, 'wb') as f:
                    pickle.dump(state, f)
            
            # Memory-efficient streaming approach
            try:
                result = running_streaming_sparse(
                    N, steps, dev, theta, initial_nodes, 
                    step_callback=save_step_callback,
                    noise_type="static_noise"
                )
                
            except MemoryError as mem_error:
                logger.error(f"Memory error in sample {sample_idx}: {mem_error}")
                raise
            except Exception as comp_error:
                logger.error(f"Computation error in sample {sample_idx}: {comp_error}")
                logger.error(traceback.format_exc())
                raise
            
            dev_computed_samples += 1
            sample_time = time.time() - sample_start_time
            
            # Force garbage collection to free memory
            gc.collect()
            
            logger.info(f"Sample {sample_idx+1}/{samples} completed in {sample_time:.1f}s")
        
        dev_time = time.time() - dev_start_time
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"min{dev[0]:.3f}_max{dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        logger.info(f"Deviation {dev_str} completed: {dev_computed_samples} samples in {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": dev_computed_samples,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        error_msg = f"Error in process for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "computed_samples": 0,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }


def compute_mean_probability_for_dev(dev_args: Tuple) -> Dict[str, Any]:
    """Worker function to compute mean probability distributions for a single deviation value in a separate process"""
    dev, process_id, N, steps, samples, source_base_dir, target_base_dir, noise_type, theta, tesselation_func = dev_args
    
    # Setup logging for this process
    dev_str = f"{dev}" if isinstance(dev, (int, float)) else f"{dev[0]}_{dev[1]}" if isinstance(dev, (tuple, list)) else str(dev)
    logger, log_file = setup_process_logging(f"meanprob_{dev_str}", process_id)
    
    try:
        logger.info(f"Starting mean probability computation for deviation {dev}")
        logger.info(f"Parameters: N={N}, steps={steps}, samples={samples}")
        
        # Import required modules
        from sqw.states import amp2prob
        from smart_loading_static import find_experiment_dir_flexible, get_experiment_dir
        
        dev_start_time = time.time()
        
        # Handle deviation format for has_noise check
        if isinstance(dev, (tuple, list)) and len(dev) == 2:
            # Direct (minVal, maxVal) format
            min_val, max_val = dev
            has_noise = max_val > 0
            dev_str = f"min{min_val:.3f}_max{max_val:.3f}"
        else:
            # Single value format
            has_noise = dev > 0
            dev_str = f"{dev:.3f}"
        
        if noise_type == "static_noise":
            noise_params = [dev]
            param_name = "static_dev"
        
        # Get source and target directories
        if noise_type == "static_noise":
            source_exp_dir, found_format = find_experiment_dir_flexible(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=source_base_dir, theta=theta)
            # For target directory (probDist), always use new structure with samples
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=target_base_dir, theta=theta, samples=samples)
        else:
            source_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=source_base_dir, theta=theta)
            target_exp_dir = get_experiment_dir(tesselation_func, has_noise, N, noise_params=noise_params, noise_type=noise_type, base_dir=target_base_dir, theta=theta, samples=samples)
        
        os.makedirs(target_exp_dir, exist_ok=True)
        logger.info(f"Processing {steps} steps for {param_name}={dev_str}")
        logger.info(f"Source: {source_exp_dir}")
        logger.info(f"Target: {target_exp_dir}")
        
        # Log initial system resources
        log_system_resources(logger, "[WORKER]")
        
        processed_steps = 0
        last_log_time = time.time()
        
        for step_idx in range(steps):
            # Check if mean probability file already exists
            mean_filename = f"mean_step_{step_idx}.pkl"
            mean_filepath = os.path.join(target_exp_dir, mean_filename)
            
            if os.path.exists(mean_filepath):
                processed_steps += 1
                continue
            
            # Log progress more frequently and monitor resources
            current_time = time.time()
            should_log_progress = (step_idx % 100 == 0)
            should_log_resources = (current_time - last_log_time >= 300)  # Every 5 minutes
            
            if should_log_progress:
                logger.info(f"Processing step {step_idx+1}/{steps} ({((step_idx+1)/steps)*100:.1f}%)")
            
            if should_log_resources:
                log_resource_usage(logger, "[PROGRESS] ")
                last_log_time = current_time
            
            step_dir = os.path.join(source_exp_dir, f"step_{step_idx}")
            
            # Optimized streaming processing - load and process samples one at a time
            mean_prob_dist = None
            valid_samples = 0
            
            for sample_idx in range(samples):
                sample_file = os.path.join(step_dir, f"sample_{sample_idx}.pkl")
                
                if not os.path.exists(sample_file):
                    continue
                
                try:
                    with open(sample_file, 'rb') as f:
                        state = pickle.load(f)
                    
                    prob_dist = amp2prob(state)
                    
                    if mean_prob_dist is None:
                        mean_prob_dist = prob_dist.copy()
                    else:
                        mean_prob_dist += prob_dist
                    
                    valid_samples += 1
                    
                    # Free memory immediately
                    del state, prob_dist
                    
                except Exception as e:
                    logger.warning(f"Could not load sample {sample_idx} for step {step_idx}: {e}")
            
            if valid_samples > 0:
                mean_prob_dist = mean_prob_dist / valid_samples
                
                # Save mean probability distribution
                with open(mean_filepath, 'wb') as f:
                    pickle.dump(mean_prob_dist, f)
                
                # Free memory
                del mean_prob_dist
            else:
                logger.warning(f"No valid samples found for step {step_idx}")
            
            processed_steps += 1
            
            # Force garbage collection periodically to keep memory usage low
            if step_idx % 50 == 0:
                gc.collect()
        
        dev_time = time.time() - dev_start_time
        logger.info(f"Deviation {dev_str} completed: {processed_steps}/{steps} steps in {dev_time:.1f}s")
        
        return {
            "dev": dev,
            "process_id": process_id,
            "processed_steps": processed_steps,
            "total_steps": steps,
            "total_time": dev_time,
            "log_file": log_file,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        # Format dev for display
        if isinstance(dev, tuple):
            dev_str = f"max{dev[0]:.3f}_min{dev[0]*dev[1]:.3f}"
        else:
            dev_str = f"{dev:.4f}"
        error_msg = f"Error in mean probability process for dev {dev_str}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "dev": dev,
            "process_id": process_id,
            "processed_steps": 0,
            "total_steps": steps,
            "total_time": 0,
            "log_file": log_file,
            "success": False,
            "error": error_msg
        }
