"""
Cluster deployment configuration and utilities.
"""

import os
import sys
import subprocess
import tarfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class ClusterConfig:
    """Configuration for cluster deployment."""
    
    # Virtual environment
    venv_name: str = "qw_venv"
    python_min_version: tuple = (3, 7)
    
    # Required packages
    required_packages: List[str] = None
    required_modules: List[str] = None
    
    # Results bundling
    results_dirs: List[str] = None
    archive_prefix: str = "experiment_results"
    use_compression: bool = False
    
    # Execution parameters
    cluster_optimized: bool = True
    N: int = 2000
    samples: int = 10
    
    def __post_init__(self):
        if self.required_packages is None:
            self.required_packages = [
                "numpy",
                "scipy", 
                "networkx",
                "matplotlib",
                "qwak-sim"
            ]
        
        if self.required_modules is None:
            self.required_modules = ["numpy", "scipy", "networkx", "matplotlib"]
        
        if self.results_dirs is None:
            self.results_dirs = ["experiments_data_samples", "experiments_data_samples_probDist"]


def run_command(cmd: str, check: bool = True, capture_output: bool = False):
    """Run a shell command and return the result."""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
        return result
    else:
        result = subprocess.run(cmd, shell=True)
        if check and result.returncode != 0:
            print(f"Command failed: {cmd}")
            sys.exit(1)
        return result


def check_python_version(min_version: tuple = (3, 7)):
    """Check if Python version is compatible."""
    if sys.version_info < min_version:
        print(f"Error: Python {'.'.join(map(str, min_version))} or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")


def setup_virtual_environment(config: ClusterConfig) -> str:
    """Create and setup virtual environment with required packages."""
    print("Setting up virtual environment...")
    
    venv_path = Path.cwd() / config.venv_name
    
    # Create virtual environment
    run_command(f"python3 -m venv {venv_path}")
    
    # Activate virtual environment and install packages
    pip_cmd = f"{venv_path}/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip")
    
    # Install required packages
    for package in config.required_packages:
        print(f"Installing {package}...")
        run_command(f"{pip_cmd} install {package}")
    
    print("Virtual environment setup complete.")
    return f"{venv_path}/bin/python"


def check_dependencies(required_modules: List[str]) -> List[str]:
    """Check if required dependencies are available."""
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return missing_modules


def bundle_results(config: ClusterConfig, N: Optional[int] = None, samples: Optional[int] = None) -> Optional[str]:
    """Bundle the results directories using tar."""
    dirs_to_bundle = []
    
    for results_dir in config.results_dirs:
        if os.path.exists(results_dir):
            dirs_to_bundle.append(results_dir)
            print(f"Found results directory: {results_dir}")
    
    if not dirs_to_bundle:
        print("Warning: No results directories found")
        return None
    
    # Create filename
    if N is not None and samples is not None:
        archive_filename = f"{config.archive_prefix}_N{N}_samples{samples}.tar"
    else:
        archive_filename = f"{config.archive_prefix}.tar"
    
    # Use compression if requested
    mode = 'w:gz' if config.use_compression else 'w'
    if config.use_compression:
        archive_filename += '.gz'
    
    try:
        # Try native tar command first (faster)
        if not config.use_compression:
            tar_cmd = f"tar -cf {archive_filename} " + " ".join(dirs_to_bundle)
            run_command(tar_cmd, check=False)
            if os.path.exists(archive_filename):
                print(f"Results bundled to {archive_filename}")
                return archive_filename
    except:
        print("tar command failed, trying Python tarfile...")
    
    try:
        # Fallback to Python tarfile module
        with tarfile.open(archive_filename, mode) as tar:
            for dir_path in dirs_to_bundle:
                tar.add(dir_path)
        print(f"Results bundled to {archive_filename}")
        return archive_filename
    except Exception as e:
        print(f"Warning: Could not create bundle: {e}")
        return None


def setup_cluster_environment(config: ClusterConfig) -> str:
    """Setup cluster environment with dependency checking and virtual environment."""
    print(f"=== Cluster Environment Setup ===")
    
    # Check Python version
    check_python_version(config.python_min_version)
    
    # Check if we need to setup virtual environment
    missing_deps = check_dependencies(config.required_modules)
    
    if missing_deps:
        print(f"Missing dependencies: {missing_deps}")
        print("Setting up virtual environment...")
        python_executable = setup_virtual_environment(config)
        return python_executable
    else:
        print("All dependencies available, using system Python")
        return sys.executable


def re_execute_with_venv(python_executable: str, script_path: str):
    """Re-execute the script with virtual environment Python."""
    print(f"Re-executing with virtual environment Python: {python_executable}")
    run_command(f"{python_executable} {script_path} --venv-ready")
