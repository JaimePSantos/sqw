"""
Cluster deployment decorator and core functionality.
"""

import sys
import os
import functools
import time
from typing import Callable, Any, Optional, Dict
from pathlib import Path

from .config import ClusterConfig, setup_cluster_environment, bundle_results, re_execute_with_venv


def cluster_deploy(
    config: Optional[ClusterConfig] = None,
    experiment_name: str = "quantum_walk",
    noise_type: str = "angle",
    **config_kwargs
):
    """
    Decorator for deploying quantum walk experiments to cluster environments.
    
    This decorator handles:
    - Python version checking
    - Dependency management 
    - Virtual environment setup
    - Experiment execution
    - Result bundling
    
    Args:
        config: ClusterConfig instance (optional, will create default if None)
        experiment_name: Name for the experiment (used in archive naming)
        noise_type: Type of noise experiment ("angle" or "tesselation_order")
        **config_kwargs: Additional configuration parameters
    
    Usage:
        @cluster_deploy(experiment_name="angle_noise", noise_type="angle")
        def run_angle_experiment():
            # Your experiment code here
            pass
    """
    def decorator(experiment_func: Callable) -> Callable:
        @functools.wraps(experiment_func)
        def wrapper(*args, **kwargs):
            # Create config if not provided
            if config is None:
                cluster_config = ClusterConfig(**config_kwargs)
                # Update archive prefix with experiment name
                cluster_config.archive_prefix = f"{experiment_name}_results"
            else:
                cluster_config = config
            
            print(f"=== {experiment_name.title()} Cluster Deployment ===")
            
            # Check for virtual environment flag
            if len(sys.argv) > 1 and sys.argv[1] == "--venv-ready":
                # We're running in virtual environment, just run experiment
                print("Running in virtual environment, executing experiment...")
                result = experiment_func(*args, **kwargs)
                print(f"=== {experiment_name.title()} Experiment completed in virtual environment ===")
                return result
            
            # Setup cluster environment
            python_executable = setup_cluster_environment(cluster_config)
            
            # If we needed to create virtual environment, re-execute
            if python_executable != sys.executable:
                script_path = sys.argv[0]  # Current script path
                re_execute_with_venv(python_executable, script_path)
                
                # Bundle results and exit
                bundle_results(cluster_config, cluster_config.N, cluster_config.samples)
                print(f"=== {experiment_name.title()} Experiment completed ===")
                return None
            
            # If we reach here, dependencies are available - run the experiment
            print("Dependencies available, running experiment...")
            result = experiment_func(*args, **kwargs)
            
            # Bundle results after successful execution
            print("Creating TAR archive of results...")
            archive_filename = bundle_results(cluster_config, cluster_config.N, cluster_config.samples)
            
            print("=== Analysis Instructions ===")
            if archive_filename:
                print(f"Results archived in: {archive_filename}")
            print("To analyze the results, transfer the tar file and extract it, then use:")
            print("- experiments_data_samples/ contains the raw quantum states for each sample")
            print("- experiments_data_samples_probDist/ contains the mean probability distributions")
            print(f"Both directories maintain the {noise_type} directory structure for easy analysis.")
            
            return result
        
        return wrapper
    return decorator


def cluster_experiment(
    N: int = 2000,
    samples: int = 10,
    experiment_name: str = "quantum_walk",
    noise_type: str = "angle"
):
    """
    Simplified cluster deployment decorator with common parameters.
    
    Args:
        N: System size for cluster-optimized parameters
        samples: Number of samples per parameter
        experiment_name: Name for the experiment
        noise_type: Type of noise experiment
    """
    config = ClusterConfig(
        N=N,
        samples=samples,
        archive_prefix=f"{experiment_name}_results"
    )
    
    return cluster_deploy(
        config=config,
        experiment_name=experiment_name,
        noise_type=noise_type
    )
