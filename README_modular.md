# Static Noise Quantum Walk Experiment - Modular Version

This is a modularized version of the static noise quantum walk experiment code. The original monolithic script has been split into focused, decoupled modules for better maintainability and reusability.

## Quick Start

Simply run the main experiment:

```bash
python main_static_experiment.py
```

## Module Structure

### Core Modules

1. **`main_static_experiment.py`** - Main entry point with simple interface
2. **`experiment_config.py`** - Configuration management and parameter validation
3. **`experiment_orchestrator.py`** - Main experiment logic and phase coordination
4. **`worker_functions.py`** - Multiprocessing worker functions for computation
5. **`data_manager.py`** - Data processing, archiving, and file management
6. **`experiment_logging.py`** - Centralized logging for master and worker processes
7. **`system_monitor.py`** - System resource monitoring and signal handling
8. **`background_executor.py`** - Background process execution utilities

### Module Responsibilities

#### `experiment_config.py`
- Configuration parameters with defaults
- Environment variable override handling
- Parameter validation and derived calculations
- Resource estimation

#### `experiment_orchestrator.py` 
- Main experiment workflow coordination
- Phase management (sample computation, analysis, plotting, archiving)
- Error handling and recovery
- Result aggregation

#### `worker_functions.py`
- `compute_dev_samples()` - Multiprocess worker for sample computation
- `compute_mean_probability_for_dev()` - Multiprocess worker for mean probability calculation
- Process-specific error handling and logging

#### `data_manager.py`
- Standard deviation data creation and loading
- Archive creation with multiprocessing support
- Mean probability distribution management
- File I/O operations

#### `experiment_logging.py`
- Master process logging setup
- Worker process logging setup  
- Structured log formatting
- Process result summarization

#### `system_monitor.py`
- System resource monitoring (CPU, memory)
- Progress tracking with ETA calculation
- Signal handling for graceful shutdown
- Resource estimation and warnings

#### `background_executor.py`
- Safe background process execution
- PID file management
- Platform-specific background handling (Windows/Unix)
- Process cleanup and error recovery

## Usage Examples

### 1. Full Pipeline (Default)
```python
from experiment_config import ExperimentConfig
from experiment_orchestrator import run_static_experiment

config = ExperimentConfig(
    N=20000,
    samples=20,
    devs=[(0, 0), (0, 0.2), (0, 0.6), (0, 0.8), (0, 1)],
    enable_plotting=True,
    create_tar_archive=False
)

results = run_static_experiment(config)
```

### 2. Quick Test
```python
config = ExperimentConfig(
    N=1000,          # Small system for testing
    samples=5,       # Few samples for speed
    devs=[(0, 0), (0, 0.2)],  # Just two noise levels
    max_processes=2  # Limited processes
)

results = run_static_experiment(config)
```

### 3. Samples Only (for cluster)
```python
config = ExperimentConfig(
    N=20000,
    samples=20,
    calculate_samples_only=True,  # Only compute samples
    skip_sample_computation=False,
    enable_plotting=False,        # No plotting needed
    create_tar_archive=False      # No archiving needed
)

results = run_static_experiment(config)
```

### 4. Analysis Only (local processing)
```python
config = ExperimentConfig(
    N=20000,
    samples=20,
    calculate_samples_only=False,   
    skip_sample_computation=True,   # Skip sample computation
    enable_plotting=True,           # Enable plotting
    create_tar_archive=True         # Create final archive
)

results = run_static_experiment(config)
```

## Configuration Options

### Core Parameters
- `N`: System size (default: 20000)
- `samples`: Samples per deviation (default: 20)  
- `theta`: Base theta parameter (default: π/3)
- `devs`: List of deviation values (default: [(0,0), (0,0.2), (0,0.6), (0,0.8), (0,1)])

### Execution Modes
- `calculate_samples_only`: Only compute samples, skip analysis
- `skip_sample_computation`: Skip sample computation, analysis only
- Default: Full pipeline (samples + analysis + plotting + archiving)

### Plotting Options
- `enable_plotting`: Enable/disable all plotting
- `use_loglog_plot`: Use log-log scale for standard deviation plots
- `plot_final_probdist`: Plot final probability distributions
- `save_figures`: Save plots to files

### Multiprocessing Options
- `max_processes`: Max processes for sample computation (None = auto)
- `use_multiprocess_mean_prob`: Enable multiprocessing for mean probability calculation
- `max_mean_prob_processes`: Max processes for mean probability (None = auto)

### Archive Options
- `create_tar_archive`: Create compressed archive of results
- `use_multiprocess_archiving`: Use multiprocessing for archiving
- `exclude_samples_from_archive`: Exclude raw samples (keep processed data only)

### Background Execution
- `run_in_background`: Run experiment in background process
- `background_log_file`: Log file for background process
- `background_pid_file`: PID file for background process tracking

## Environment Variable Overrides

All configuration options can be overridden via environment variables:

```bash
export ENABLE_PLOTTING=false
export CREATE_TAR_ARCHIVE=true
export MAX_PROCESSES=4
export SKIP_SAMPLE_COMPUTATION=true
python main_static_experiment.py
```

## Output Structure

```
experiments_data_samples/          # Raw sample data
experiments_data_samples_probDist/ # Mean probability distributions  
experiments_data_samples_std/      # Standard deviation data
process_logs/                      # Individual process logs
static_experiment_multiprocess.log # Master process log
*.png                             # Generated plots
experiments_data_N*_samples*.tar.gz # Optional archives
```

## Key Improvements

1. **Modular Design**: Clear separation of concerns across focused modules
2. **Simple Interface**: Main function takes only the parameters you want to control
3. **Decoupled Components**: Each module has minimal dependencies
4. **Configuration Management**: Centralized, validated configuration with environment overrides
5. **Better Error Handling**: Module-specific error handling and recovery
6. **Improved Logging**: Structured logging with process-specific log files
7. **Resource Management**: Better system monitoring and resource estimation
8. **Flexible Execution**: Easy switching between different execution modes

## Migration from Original Code

The original `static_local_logged_mp.py` functionality is preserved but split across modules:

- Configuration → `experiment_config.py`
- Signal handling → `system_monitor.py`  
- Logging → `experiment_logging.py`
- Workers → `worker_functions.py`
- Data management → `data_manager.py`
- Background execution → `background_executor.py`
- Main logic → `experiment_orchestrator.py`
- User interface → `main_static_experiment.py`

All original features are maintained including multiprocessing, timeout handling, resource monitoring, graceful shutdown, and background execution.
