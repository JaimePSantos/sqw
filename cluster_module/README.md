# Cluster Deployment Module

A comprehensive module for deploying quantum walk experiments to cluster environments with automatic dependency management, virtual environment setup, and result bundling.

## Features

- **Automatic Dependency Management**: Checks for required Python packages and installs them if missing
- **Virtual Environment Setup**: Creates isolated environments with all necessary dependencies
- **Result Bundling**: Automatically packages experiment results for easy transfer
- **Progress Monitoring**: Detailed progress tracking and timing information
- **Cross-Platform Compatibility**: Works on both Linux clusters and Windows development environments

## Quick Start

### Using the Decorator

```python
from cluster_module import cluster_deploy

@cluster_deploy(
    experiment_name="my_experiment",
    noise_type="angle",
    N=2000,
    samples=10
)
def my_experiment():
    # Your experiment code here
    pass

if __name__ == "__main__":
    my_experiment()
```

### Using the Simplified Decorator

```python
from cluster_module import cluster_experiment

@cluster_experiment(N=1000, samples=5, experiment_name="test")
def my_test_experiment():
    # Your experiment code here
    pass
```

## Configuration

### ClusterConfig Options

```python
from cluster_module import ClusterConfig, cluster_deploy

config = ClusterConfig(
    # Virtual environment
    venv_name="my_venv",
    python_min_version=(3, 7),
    
    # Required packages
    required_packages=["numpy", "scipy", "networkx", "matplotlib", "qwak-sim"],
    required_modules=["numpy", "scipy", "networkx", "matplotlib"],
    
    # Results bundling
    results_dirs=["experiments_data_samples", "experiments_data_samples_probDist"],
    archive_prefix="my_experiment_results",
    use_compression=False,  # False for faster bundling
    
    # Execution parameters
    N=2000,
    samples=10
)

@cluster_deploy(config=config, experiment_name="my_experiment")
def my_experiment():
    pass
```

## How It Works

### 1. Environment Setup
The decorator automatically:
- Checks Python version compatibility
- Verifies required dependencies are installed
- Creates virtual environment if dependencies are missing
- Installs required packages in the virtual environment

### 2. Execution Management
- Re-executes the script with virtual environment Python if needed
- Handles the `--venv-ready` flag to avoid infinite recursion
- Manages experiment execution flow

### 3. Result Bundling
After successful execution:
- Bundles specified result directories into a single TAR archive
- Uses native `tar` command for speed (falls back to Python `tarfile` if needed)
- Names archives with experiment parameters for easy identification

## Directory Structure

```
cluster_module/
├── __init__.py              # Module exports
├── config.py               # Configuration and utility functions  
└── cluster_deployment.py   # Main decorator implementation
```

## Usage Examples

### Angle Noise Experiment

```python
#!/usr/bin/env python3
from cluster_module import cluster_deploy

@cluster_deploy(
    experiment_name="angle_noise",
    noise_type="angle",
    N=2000,
    samples=10
)
def run_angle_experiment():
    # Import after cluster setup
    import numpy as np
    from sqw.tesselations import even_line_two_tesselation
    # ... experiment code ...
    return results

if __name__ == "__main__":
    run_angle_experiment()
```

### Tesselation Order Experiment

```python
#!/usr/bin/env python3
from cluster_module import cluster_deploy

@cluster_deploy(
    experiment_name="tesselation_order", 
    noise_type="tesselation_order",
    N=2000,
    samples=10
)
def run_tesselation_experiment():
    # Import after cluster setup
    import numpy as np
    from sqw.utils import tesselation_choice
    # ... experiment code ...
    return results

if __name__ == "__main__":
    run_tesselation_experiment()
```

## Deployment Workflow

### 1. Local Development
- Write experiment using the `@cluster_deploy` decorator
- Test locally (decorator will use system Python if dependencies available)

### 2. Cluster Deployment
- Copy script to cluster
- Run: `python3 my_experiment.py`
- Decorator automatically handles virtual environment setup if needed
- Results are bundled into TAR archive for download

### 3. Result Transfer
- Download the generated TAR file (e.g., `angle_noise_results_N2000_samples10.tar`)
- Extract locally: `tar -xf angle_noise_results_N2000_samples10.tar`
- Analyze results using local tools

## Advanced Features

### Custom Configuration
```python
config = ClusterConfig(
    venv_name="custom_venv",
    required_packages=["numpy", "custom-package"],
    archive_prefix="custom_results",
    use_compression=True  # Enable compression for smaller files
)

@cluster_deploy(config=config)
def my_experiment():
    pass
```

### Multiple Result Directories
```python
config = ClusterConfig(
    results_dirs=[
        "experiments_data_samples",
        "experiments_data_samples_probDist",
        "additional_analysis",
        "plots"
    ]
)
```

## Benefits

### Code Quality
- **Eliminates Duplication**: No more copying cluster setup code between experiments
- **Consistent Interface**: Same decorator pattern for all experiment types
- **Easy Maintenance**: Cluster logic centralized in one module
- **Better Testing**: Decorator can be easily tested and mocked

### Deployment Efficiency  
- **Automatic Setup**: No manual virtual environment management
- **Dependency Checking**: Prevents runtime errors from missing packages
- **Result Bundling**: Single file for easy transfer from cluster
- **Progress Tracking**: Clear feedback during long-running experiments

### User Experience
- **Simple Usage**: Just add decorator to existing experiment functions
- **Flexible Configuration**: Override defaults as needed
- **Cross-Platform**: Same code works on Linux clusters and Windows development
- **Self-Contained**: Scripts include all necessary cluster management logic

## Migration from Old Cluster Files

### Before (Old Cluster Files)
```python
# 200+ lines of duplicate cluster management code
def run_command(cmd): ...
def check_python_version(): ...
def setup_virtual_environment(): ...
def check_dependencies(): ...
def zip_results(): ...
def main(): ...
def run_experiment(): ...
```

### After (New Decorator Approach)
```python
from cluster_module import cluster_deploy

@cluster_deploy(experiment_name="my_experiment")
def run_experiment():
    # Just the experiment logic
    pass

if __name__ == "__main__":
    run_experiment()
```

**Result**: ~80% code reduction with better maintainability and consistency.
