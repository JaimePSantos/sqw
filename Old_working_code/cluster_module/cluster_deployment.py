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
    venv_name: str = "qw_venv",
    check_existing_env: bool = True,
    create_tar_archive: bool = True,
    use_compression: bool = False,
    **config_kwargs
):
    """
    Decorator for deploying quantum walk experiments to cluster environments.
    
    This decorator handles:
    - Python version checking
    - Dependency management 
    - Virtual environment setup (with existing environment checking)
    - Experiment execution
    - Result bundling (optional)
    
    Args:
        config: ClusterConfig instance (optional, will create default if None)
        experiment_name: Name for the experiment (used in archive naming)
        noise_type: Type of noise experiment ("angle" or "tesselation_order")
        venv_name: Name of the virtual environment to create/check
        check_existing_env: Whether to check for and reuse existing virtual environments
        create_tar_archive: Whether to create tar archives of results
        use_compression: Whether to compress tar archives (gzip)
        **config_kwargs: Additional configuration parameters
    
    Usage:
        @cluster_deploy(
            experiment_name="angle_noise", 
            noise_type="angle",
            venv_name="my_custom_env",
            create_tar_archive=False
        )
        def run_angle_experiment():
            # Your experiment code here
            pass
    """
    def decorator(experiment_func: Callable) -> Callable:
        @functools.wraps(experiment_func)
        def wrapper(*args, **kwargs):
            # Create config if not provided
            if config is None:
                cluster_config = ClusterConfig(
                    venv_name=venv_name,
                    check_existing_env=check_existing_env,
                    create_tar_archive=create_tar_archive,
                    use_compression=use_compression,
                    **config_kwargs
                )
                # Update archive prefix with experiment name
                cluster_config.archive_prefix = f"{experiment_name}_results"
            else:
                cluster_config = config
            
            print(f"=== {experiment_name.title()} Cluster Deployment ===")
            print(f"Virtual environment: {cluster_config.venv_name}")
            print(f"Check existing env: {cluster_config.check_existing_env}")
            print(f"Create TAR archive: {cluster_config.create_tar_archive}")
            if cluster_config.create_tar_archive:
                print(f"Use compression: {cluster_config.use_compression}")
            
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
                if cluster_config.create_tar_archive:
                    print("Creating TAR archive of results...")
                    bundle_results(cluster_config, cluster_config.N, cluster_config.samples)
                else:
                    bundle_results(cluster_config, cluster_config.N, cluster_config.samples)  # Will just report directories
                print(f"=== {experiment_name.title()} Experiment completed ===")
                return None
            
            # If we reach here, dependencies are available - run the experiment
            print("Dependencies available, running experiment...")
            result = experiment_func(*args, **kwargs)
            
            # Bundle results after successful execution
            if cluster_config.create_tar_archive:
                print("Creating TAR archive of results...")
                archive_filename = bundle_results(cluster_config, cluster_config.N, cluster_config.samples)
            else:
                archive_filename = bundle_results(cluster_config, cluster_config.N, cluster_config.samples)  # Will just report directories
            
            print("=== Analysis Instructions ===")
            if archive_filename:
                print(f"Results archived in: {archive_filename}")
                print("To analyze the results, transfer the tar file and extract it, then use:")
            else:
                print("Results are available in the following directories:")
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
    noise_type: str = "angle",
    venv_name: str = "qw_venv",
    check_existing_env: bool = True,
    create_tar_archive: bool = True,
    use_compression: bool = False
):
    """
    Simplified cluster deployment decorator with common parameters.
    
    Args:
        N: System size for cluster-optimized parameters
        samples: Number of samples per parameter
        experiment_name: Name for the experiment
        noise_type: Type of noise experiment
        venv_name: Name of the virtual environment to create/check
        check_existing_env: Whether to check for and reuse existing virtual environments
        create_tar_archive: Whether to create tar archives of results
        use_compression: Whether to compress tar archives (gzip)
    """
    config = ClusterConfig(
        N=N,
        samples=samples,
        venv_name=venv_name,
        check_existing_env=check_existing_env,
        create_tar_archive=create_tar_archive,
        use_compression=use_compression,
        archive_prefix=f"{experiment_name}_results"
    )
    
    return cluster_deploy(
        config=config,
        experiment_name=experiment_name,
        noise_type=noise_type
    )
