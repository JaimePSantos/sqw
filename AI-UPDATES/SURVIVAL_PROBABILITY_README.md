# Survival Probability Analysis

This directory contains scripts for calculating and analyzing survival probabilities from quantum walk experiments.

## Overview

The survival probability is calculated as the sum of probabilities within a specified range [fromNode, toNode] at each time step. This is useful for analyzing localization effects and probability distribution spreading in quantum walks.

## Scripts

### 1. `survival_probability_analysis.py`
Main script that calculates survival probabilities from existing probability distributions.

**Features:**
- Loads probability distributions from `experiments_data_samples_probDist`
- Calculates survival probabilities for configurable node ranges
- Saves results to `experiments_data_samples_survivalProb`
- Supports multiple range definitions (center nodes, halves, etc.)
- Memory-efficient processing with progress tracking

**Usage:**
```bash
python survival_probability_analysis.py
```

**Prerequisites:**
- Run the cluster experiment first to generate probability distributions
- QWAK library must be accessible (for ProbabilityDistribution class)

### 2. `survival_probability_loader.py`
Utility script for loading and visualizing survival probability results.

**Features:**
- Load survival probability data and metadata
- Generate plots comparing different noise levels
- Compare different survival ranges
- Calculate summary statistics

**Usage:**
```bash
python survival_probability_loader.py
```

## Configuration

### Survival Ranges
The analysis calculates survival probabilities for several predefined ranges:

- `center_single`: Single center node only
- `center_5`: 5 nodes around center (center ± 2)
- `center_11`: 11 nodes around center (center ± 5)
- `center_21`: 21 nodes around center (center ± 10)
- `left_half`: Left half of the system (0 to center)
- `right_half`: Right half of the system (center to N-1)

### Node Position Specifications
The scripts support flexible node position specifications:
- Integer values: Direct node indices
- `"center"`: Center node (N//2)
- `"center+X"`: Center node plus offset
- `"center-X"`: Center node minus offset
- `"N-X"`: System size minus offset

## Data Structure

### Input Data
The script expects probability distributions in the format:
```
experiments_data_samples_probDist/
└── dummy_tesselation_func_static_noise/
    └── theta_X.XXXXXX/
        └── dev_minX.XXX_maxX.XXX/
            └── N_XXXXX/
                ├── mean_step_0.pkl
                ├── mean_step_1.pkl
                └── ...
```

### Output Data
Survival probabilities are saved in a similar structure:
```
experiments_data_samples_survivalProb/
├── dummy_tesselation_func_static_noise/
│   └── theta_X.XXXXXX/
│       └── dev_minX.XXX_maxX.XXX/
│           └── N_XXXXX/
│               ├── survival_center_single.pkl
│               ├── survival_center_5.pkl
│               ├── survival_center_11.pkl
│               ├── survival_center_21.pkl
│               ├── survival_left_half.pkl
│               └── survival_right_half.pkl
└── summary/
    ├── metadata.pkl
    └── all_survival_probabilities.pkl
```

## Example Usage

1. **Run the analysis:**
   ```bash
   python survival_probability_analysis.py
   ```

2. **Load and visualize results:**
   ```bash
   python survival_probability_loader.py
   ```

3. **Load specific data in your own script:**
   ```python
   import pickle
   
   # Load metadata
   with open('experiments_data_samples_survivalProb/summary/metadata.pkl', 'rb') as f:
       metadata = pickle.load(f)
   
   # Load all data
   with open('experiments_data_samples_survivalProb/summary/all_survival_probabilities.pkl', 'rb') as f:
       all_data = pickle.load(f)
   
   # Access survival probability for first deviation, center_11 range
   survival_probs = all_data[0]['survival_data']['center_11']
   ```

## Implementation Details

### Survival Probability Calculation
The survival probability is calculated using the implementation from the QWAK ProbabilityDistribution class:

```python
def survivalProb(self, fromNode, toNode) -> float:
    survProb = 0
    if fromNode == toNode:
        return self._probVec[int(fromNode)]
    else:
        for i in range(int(fromNode), int(toNode) + 1):
            survProb += self._probVec[i]
    return survProb
```

### Error Handling
- Corrupted probability data is handled gracefully (None values)
- Missing files are reported with clear error messages
- Progress tracking and logging for long computations

### Memory Efficiency
- Processes one deviation at a time to minimize memory usage
- Streams probability distributions rather than loading all at once
- Saves individual range data separately for flexible loading

## Configuration Parameters

Key parameters that can be modified in `survival_probability_analysis.py`:

```python
# System parameters (should match original experiment)
N = 20000
steps = N//4
theta = math.pi/3

# Deviation values (should match original experiment)
devs = [
    (0,0),      # No noise
    (0, 0.2),   # Small noise
    (0, 0.6),   # Medium noise
    (0, 0.8),   # Medium noise
    (0, 1),     # High noise
]

# Survival ranges (can be customized)
SURVIVAL_RANGES = [
    {"name": "center_single", "from_node": "center", "to_node": "center"},
    {"name": "center_5", "from_node": "center-2", "to_node": "center+2"},
    # ... add more ranges as needed
]
```

## Output Files

- `survival_probability_analysis.log`: Detailed execution log
- `experiments_data_samples_survivalProb/`: Main results directory
- `survival_probability_figures/`: Generated plots (from loader script)

## Dependencies

- numpy
- pickle (built-in)
- matplotlib (for visualization)
- QWAK library (for ProbabilityDistribution class)
- Smart loading functions from the sqw project
