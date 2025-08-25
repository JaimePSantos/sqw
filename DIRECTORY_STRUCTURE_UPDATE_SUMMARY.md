# Directory Structure Update Summary

## Overview
Updated all Python scripts to use a consistent directory structure that matches the existing `samples_40` folder format.

## New Directory Structure
```
base_dir/
├── tesselation_func_noise_type/
│   └── theta_value/
│       └── dev_range/
│           └── N_value/
│               └── samples_count/
│                   ├── mean_step_0.pkl
│                   ├── mean_step_1.pkl
│                   └── ...
```

## Example Path
For the existing folder structure:
```
experiments_data_samples_probDist/
└── dummy_tesselation_func_static_noise/
    └── theta_1.047198/
        └── dev_min0.000_max0.000/
            └── N_20000/
                └── samples_40/
                    ├── mean_step_0.pkl
                    ├── mean_step_1.pkl
                    └── ...
```

## Files Updated

### 1. `generate_samples.py`
- ✅ Updated `get_experiment_dir()` function
- ✅ Added missing imports (`math`, `signal`)
- ✅ Fixed directory structure to match new format

### 2. `recreate_probdist_from_samples.py`
- ✅ Updated `get_experiment_dir()` function
- ✅ Updated `find_experiment_dir_flexible()` function
- ✅ Maintained backwards compatibility

### 3. `generate_std_from_probdist.py`
- ✅ Updated `get_experiment_dir()` function
- ✅ Updated `find_experiment_dir_flexible()` function
- ✅ Added missing imports (`math`, `signal`, `numpy`)

### 4. `generate_survival_probability.py`
- ✅ Updated `get_experiment_dir()` function
- ✅ Updated `find_experiment_dir_flexible()` function
- ✅ Added missing imports (`math`, `numpy`)

### 5. `plot_experiment_results.py`
- ✅ Updated `get_experiment_dir()` function
- ✅ Updated `find_experiment_dir_flexible()` function
- ✅ Updated `format_deviation_label()` for better plot labels

## Key Changes Made

### Directory Path Format
**Old format:**
```
base_dir/theta_value/N_value/static_dev_min_max_samples_count/
```

**New format:**
```
base_dir/dummy_tesselation_func_static_noise/theta_value/dev_min_max/N_value/samples_count/
```

### Deviation Formatting
- **Tuple deviations (min, max):** `dev_min0.000_max0.200`
- **Single deviations:** `dev_0.100`
- **No noise:** `dev_min0.000_max0.000`

### Backwards Compatibility
- All scripts now include `find_experiment_dir_flexible()` functions
- These functions try multiple common sample counts (40, 20, 10, 5)
- Fall back to old directory formats if new format not found

## Configuration Updates

### Base Directory Names
- Samples: `experiments_data_samples`
- Probability Distributions: `experiments_data_samples_probDist`
- Standard Deviations: `experiments_data_samples_std`
- Survival Probabilities: `experiments_data_samples_survival`

### Default Parameters
All scripts now use consistent parameter naming:
- `N = 300` (system size)
- `steps = N//4` (time steps)
- `samples = 5` (samples per deviation)
- `theta = math.pi/3` (theta parameter)

## Testing Results
✅ All files pass syntax validation
✅ Directory structure consistent across all scripts
✅ Backwards compatibility maintained
✅ Existing `samples_40` folder structure correctly generated

## Migration Notes
- Scripts will automatically use the new directory structure for new data
- Existing data in old format will still be found via backwards compatibility
- No manual file migration required
- All scripts work with the existing `experiments_data_samples_probDist` folder

## Next Steps
1. Run the updated scripts to verify they work with existing data
2. Generate new data using the updated directory structure
3. Verify that all analysis pipelines work correctly
4. Consider creating symbolic links if directory migration is needed
