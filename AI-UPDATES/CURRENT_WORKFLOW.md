# Current Active Workflow

## Quick Start Guide

This repository now follows a modular pipeline approach for quantum walk experiments.

### Essential Scripts (In Order)

1. **`generate_samples.py`** - Generate raw sample data
   ```bash
   python generate_samples.py
   ```

2. **`generate_probdist_from_samples.py`** - Convert samples to probability distributions
   ```bash
   python generate_probdist_from_samples.py
   ```

3. **`generate_std_from_probdist.py`** - Calculate standard deviations
   ```bash
   python generate_std_from_probdist.py
   ```

4. **`generate_survival_probability.py`** - Calculate survival probabilities
   ```bash
   python generate_survival_probability.py
   ```

5. **`static_cluster_plot_only.py`** - Generate all plots
   ```bash
   python static_cluster_plot_only.py
   ```

### Alternative Plotting
- **`plot_experiment_results.py`** - Comprehensive plotting with more options

### Background Execution
- **`safe_background_launcher.py`** - Run experiments in background (Linux/cluster)

### Core Library
- **`sqw/`** - Main quantum walk library
- **`jaime_scripts.py`** - Utility functions
- **`smart_loading_static.py`** - Data loading functions

### Development
- **`generalised_sqw.ipynb`** - Interactive development notebook

## Configuration

Edit the parameter sections in each script:
- System size: `N`
- Time steps: `steps`
- Samples per deviation: `samples`
- Deviation values: `devs`
- Theta parameter: `theta`

## Data Directories

- `experiments_data_samples/` - Raw sample files
- `experiments_data_samples_probDist/` - Probability distributions
- `experiments_data_samples_std/` - Standard deviation data
- `experiments_data_samples_survival/` - Survival probability data
- `survival_probability_figures/` - Generated plots
- `logs/` - Execution logs

## Legacy Code

All deprecated and legacy scripts have been moved to `Old_working_code/`. See `Old_working_code/MIGRATION_SUMMARY.md` for details.
