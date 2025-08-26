# Migration Summary - Code Organization

## Date: August 25, 2025

This document summarizes the file reorganization to distinguish between active/essential code and legacy/deprecated code.

## ESSENTIAL FILES KEPT IN ROOT DIRECTORY

### Core Library
- `sqw/` - Main quantum walk library (all files)
- `jaime_scripts.py` - Central utility functions and experiment infrastructure
- `smart_loading_static.py` - Data loading and smart caching functions

### Active Generation Pipeline
- `generate_samples.py` - Primary sample generation script (multiprocessing)
- `generate_probdist_from_samples.py` - Probability distribution generation
- `generate_std_from_probdist.py` - Standard deviation calculation
- `generate_survival_probability.py` - Survival probability generation

### Plotting and Analysis
- `static_cluster_plot_only.py` - Current optimized plotting interface
- `plot_experiment_results.py` - Comprehensive plotting utilities
- `survival_probability_loader.py` - Survival probability data management

### Active Entry Points
- `main_static_experiment.py` - Main experiment interface (needs module fix)
- `example_usage.py` - Usage examples (needs module fix)

### Infrastructure and Utilities
- `background_launcher.py` - Background process management
- `bg_manager.py` - Background manager utilities
- `safe_background_launcher.py` - Safe background execution
- `smart_loading.py` - General data loading utilities

### Active Development
- `generalised_sqw.ipynb` - Active development notebook

### Documentation and Configuration
- `README.md` - Project documentation
- `CHANGELOG.md` - Version history
- `.github/`, `.gitignore`, etc. - Git and CI configuration

## FILES MOVED TO Old_Working_Code/

### legacy_experiments/
These are superseded by the new generation pipeline:
- `static_cluster_logged_mp.py` - Legacy cluster experiment (superseded by generate_*.py pipeline)
- `static_local_logged_mp.py` - Legacy local experiment (superseded by generate_*.py pipeline)
- `angle_cluster_logged.py` - Legacy angle cluster experiment
- `tesselation_cluster_clean.py` - Legacy tesselation experiment
- `tesselation_cluster_logged.py` - Legacy tesselation logged experiment
- `StaggeredQW_basic.py` - Legacy basic staggered QW implementation
- `StaggeredQW_static_noise.py` - Legacy static noise implementation

### deprecated_scripts/
These are old utilities that are no longer needed:
- `states_to_probabilities_mp.py` - Superseded by generate_probdist_from_samples.py
- `config_examples.py` - Old configuration examples
- `mode_switcher.py` - Deprecated mode switching utility

### test_analysis_scripts/
These are analysis and testing scripts not part of main pipeline:
- `analyze_cluster_data.py` - Cluster data analysis utility
- `survival_probability_analysis.py` - Survival probability analysis utility
- `diagnostic_spreading_test.py` - Diagnostic testing script
- `demo_samples_structure.py` - Sample structure demonstration

## CURRENT ACTIVE WORKFLOW

The current recommended workflow is:

1. **Sample Generation**: `python generate_samples.py`
2. **Probability Distributions**: `python generate_probdist_from_samples.py`
3. **Standard Deviations**: `python generate_std_from_probdist.py`
4. **Survival Probabilities**: `python generate_survival_probability.py`
5. **Plotting**: `python static_cluster_plot_only.py` or `python plot_experiment_results.py`

## NOTES

- The `main_static_experiment.py` and `example_usage.py` reference missing `static_experiment_modules` - these may need to be fixed or replaced
- All moved files are preserved and can be restored if needed
- The `sqw/` core library remains untouched and fully functional
- Data directories (`experiments_data_*`, `logs/`, etc.) are preserved
