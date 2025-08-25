# Quantum Walk Sample Generation and Probability Distribution Recreation

This repository contains two focused scripts for managing quantum walk experiment data:

## Scripts Overview

### 1. `generate_samples.py`
**Purpose**: Generate sample files for quantum walk experiments with multiprocessing support.

**Key Features**:
- Multi-process execution (one process per deviation value)
- Memory-efficient sparse matrix computation
- Comprehensive logging for each process
- Smart skipping of existing samples
- Directory structure management with N, steps, samples, and theta tracking

### 2. `recreate_probdist_from_samples.py` 
**Purpose**: Recreate probability distribution files from existing sample data.

**Key Features**:
- Multi-process execution (one process per deviation value)
- Smart file validation (checks if probDist files exist and have valid data)
- Automatic directory structure handling
- Comprehensive logging and error recovery
- Skips valid existing files, recreates missing/corrupted ones

## Quick Start

### Step 1: Configure Parameters

Edit the configuration section in both scripts to match your experiment:

```python
# Experiment parameters
N = 100                # System size
steps = N//4           # Time steps
samples = 5            # Samples per deviation
theta = math.pi/3      # Theta parameter for static noise

# Deviation values
devs = [
    (0,0),              # No noise
    (0, 0.2),           # Small noise range
    (0, 0.5),           # Medium noise range  
]
```

### Step 2: Generate Sample Files

```bash
python generate_samples.py
```

This will create sample files in the `experiments_data_samples` directory structure:
```
experiments_data_samples/
├── theta_1.047198/
│   └── N_100/
│       ├── static_dev_min0.000_max0.000_samples_5/
│       │   ├── step_0/
│       │   │   ├── sample_0.pkl
│       │   │   ├── sample_1.pkl
│       │   │   └── ...
│       │   └── step_1/
│       │       └── ...
│       └── ...
```

### Step 3: Create Probability Distributions

```bash
python recreate_probdist_from_samples.py
```

This will create probability distribution files in the `experiments_data_samples_probDist` directory:
```
experiments_data_samples_probDist/
├── theta_1.047198/
│   └── N_100/
│       └── static_dev_min0.000_max0.000_samples_5/
│           ├── mean_step_0.pkl
│           ├── mean_step_1.pkl
│           └── ...
```

## Directory Structure

The scripts automatically manage directory structures that include:
- **Theta parameter**: `theta_{value}` folders for different theta values
- **System size**: `N_{size}` folders for different system sizes
- **Deviation parameters**: `static_dev_min{min}_max{max}_samples_{count}` folders
- **Step organization**: `step_{index}` folders containing sample files
- **Sample tracking**: Individual `sample_{index}.pkl` files for each sample

## Multiprocessing

Both scripts use multiprocessing for efficiency:
- **One process per deviation value** allows parallel computation
- **Conservative resource usage** to avoid system overload
- **Comprehensive logging** for each process with individual log files
- **Graceful error handling** with detailed error reporting

## Logging

Each script creates detailed logs:

### Sample Generation Logs:
- `sample_generation_master.log` - Master process log
- `sample_generation_logs/process_dev_{deviation}_samples.log` - Individual process logs

### Probability Distribution Recreation Logs:
- `recreate_probDist/probdist_recreation_master.log` - Master process log  
- `recreate_probDist/process_dev_{deviation}_probdist.log` - Individual process logs

## Error Recovery

The scripts are designed for robustness:

1. **Automatic resumption**: If a script is interrupted, re-running will skip completed work
2. **File validation**: Checks for existing valid files before regenerating
3. **Memory management**: Uses sparse matrices and garbage collection for large problems
4. **Timeout handling**: Configurable timeouts for long-running processes

## Configuration Options

### Key Parameters to Adjust:

```python
# System parameters
N = 100                # System size (adjust based on computational resources)
steps = N//4           # Time steps (adjust for desired simulation length)
samples = 5            # Samples per deviation (more samples = better statistics)
theta = math.pi/3      # Quantum walk parameter

# Deviation values (customize for your experiment)
devs = [
    (0,0),              # Perfect evolution (no noise)
    (0, 0.2),           # Small noise: range [0, 0.2]
    (0, 0.5),           # Medium noise: range [0, 0.5]
]

# Performance settings
MAX_PROCESSES = min(len(devs), mp.cpu_count())  # Max parallel processes
PROCESS_TIMEOUT = 3600  # Timeout per process in seconds
```

## Performance Notes

- **Small test case**: N=100, steps=25, samples=5 runs in ~2 seconds
- **Memory usage**: Sparse matrices minimize memory requirements
- **Scaling**: Time complexity scales roughly as O(N × steps × samples)
- **Parallelization**: Each deviation runs independently, enabling efficient multiprocessing

## Example Workflow

1. **Test with small parameters** (N=100, samples=5) to verify setup
2. **Generate samples** for your experiment parameters
3. **Create probability distributions** from the samples
4. **Verify output** by checking generated directory structures
5. **Scale up** to production parameters as needed

## Troubleshooting

### Common Issues:

1. **Missing sample directories**: Ensure `generate_samples.py` completed successfully
2. **Memory errors**: Reduce N or number of parallel processes
3. **Import errors**: Ensure the `sqw` package is properly installed
4. **Directory not found**: Check that theta and parameter values match between scripts

### Debugging:

1. Check the master log file for overall progress
2. Check individual process logs for detailed error information
3. Verify directory structure matches expected format
4. Ensure Python environment has required packages (`numpy`, `scipy`, `pickle`)

This setup provides a robust, scalable foundation for quantum walk experiments with proper data management and error recovery.
