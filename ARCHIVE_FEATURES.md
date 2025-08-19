# Computation and Archive Control Features

## Overview
Added granular control for both computation phases and archiving with explicit switches that allow you to select what to compute (samples, probability distributions, standard deviations) and what to archive.

## New Computation Control Variables

### In `static_cluster_logged.py`:

#### High-Level Control (mutually exclusive)
```python
CALCULATE_SAMPLES_ONLY = True   # Only compute raw samples, skip analysis
SKIP_SAMPLE_COMPUTATION = False # Skip samples, only do analysis from existing data
```

#### Detailed Computation Control (Full Pipeline mode)
```python
COMPUTE_RAW_SAMPLES = True      # Compute quantum walk samples 
COMPUTE_PROBDIST = True         # Compute probability distributions from samples
COMPUTE_STD_DATA = True         # Compute standard deviation data from probability distributions
```

#### Archive Control
```python
CREATE_TAR_ARCHIVE = True       # Master switch for archiving
ARCHIVE_SAMPLES = True          # Include experiments_data_samples folder
ARCHIVE_PROBDIST = True         # Include experiments_data_samples_probDist folder
```

## Computation Modes

### 1. Full Pipeline (Default)
```python
CALCULATE_SAMPLES_ONLY = False
SKIP_SAMPLE_COMPUTATION = False
COMPUTE_RAW_SAMPLES = True
COMPUTE_PROBDIST = True
COMPUTE_STD_DATA = True
```
- Computes everything: samples → probability distributions → standard deviations
- Archives both samples and probability distributions

### 2. Samples Only
```python
CALCULATE_SAMPLES_ONLY = True
# Other detailed switches automatically overridden
```
- Only computes raw quantum walk samples
- Archives only samples data
- Ideal for cluster computation

### 3. Analysis Only  
```python
SKIP_SAMPLE_COMPUTATION = True
COMPUTE_PROBDIST = True     # Can be customized
COMPUTE_STD_DATA = True     # Can be customized
```
- Skips raw sample computation
- Allows selective analysis phases
- Archives only probability distributions

### 4. Probability Distributions Only
```python
CALCULATE_SAMPLES_ONLY = False
SKIP_SAMPLE_COMPUTATION = False
COMPUTE_RAW_SAMPLES = False
COMPUTE_PROBDIST = True
COMPUTE_STD_DATA = False
```
- Only computes probability distributions from existing samples
- Archives probability distributions

### 5. Standard Deviation Only
```python
CALCULATE_SAMPLES_ONLY = False
SKIP_SAMPLE_COMPUTATION = False
COMPUTE_RAW_SAMPLES = False
COMPUTE_PROBDIST = False
COMPUTE_STD_DATA = True
```
- Only computes standard deviations from existing probability distributions
- No archiving (just plotting)

## Launcher Mode Integration

The `safe_background_launcher.py` has been enhanced with new modes:

### Standard Modes
- **full**: All computation + archive both
- **samples**: Only samples computation + archive samples  
- **analysis**: All analysis phases + archive probdist
- **headless**: All computation, no plotting + archive both
- **quick**: Analysis only, no archiving

### New Detailed Modes  
- **probdist**: Only probability distribution computation + archive probdist
- **stddata**: Only standard deviation computation + plotting, no archive

## Key Features

### 1. Automatic Override Logic
High-level modes automatically override detailed switches for consistency:
- `CALCULATE_SAMPLES_ONLY=True` forces `COMPUTE_RAW_SAMPLES=True, COMPUTE_PROBDIST=False, COMPUTE_STD_DATA=False`
- `SKIP_SAMPLE_COMPUTATION=True` forces `COMPUTE_RAW_SAMPLES=False`

### 2. Parameter-Specific Filtering
Only archives folders containing the current N value, keeping archives focused on current experiment.

### 3. Smart Naming
Archive filenames reflect content:
- `experiments_data_samples_N20000_samples5_timestamp.tar.gz` (samples only)
- `experiments_data_probdist_N20000_samples5_timestamp.tar.gz` (probdist only)  
- `experiments_data_samples_probdist_N20000_samples5_timestamp.tar.gz` (both)

### 4. Comprehensive Validation
- Warns if archiving enabled but no content selected
- Prevents conflicting high-level modes
- Clear feedback on what's being computed/skipped

## Example Usage Scenarios

### Cluster Workflow
```bash
# On cluster: compute samples only
python safe_background_launcher.py samples

# Locally: analyze samples and create plots  
python safe_background_launcher.py analysis
```

### Iterative Analysis
```bash
# Compute probability distributions from existing samples
python safe_background_launcher.py probdist

# Later: add standard deviation analysis with different parameters
python safe_background_launcher.py stddata --no-background
```

### Custom Configuration
Edit switches directly in `static_cluster_logged.py`:
```python
# Custom: compute probdist and archive both for comparison
CALCULATE_SAMPLES_ONLY = False
SKIP_SAMPLE_COMPUTATION = False
COMPUTE_RAW_SAMPLES = False
COMPUTE_PROBDIST = True
COMPUTE_STD_DATA = False
ARCHIVE_SAMPLES = True      # Archive existing samples too
ARCHIVE_PROBDIST = True     # Archive new probdist
```

## Validation and Error Handling

The system validates configuration and provides clear feedback:
- Conflicting high-level modes trigger errors
- Missing dependencies for analysis phases show warnings  
- Archive content selection validated
- Detailed computation progress displayed

## Environment Variable Support

All switches support environment variable overrides from the launcher:
- `COMPUTE_RAW_SAMPLES`, `COMPUTE_PROBDIST`, `COMPUTE_STD_DATA`
- `ARCHIVE_SAMPLES`, `ARCHIVE_PROBDIST`
- Enables launcher modes to customize behavior without code changes

This flexible system allows efficient workflow management for both cluster computing and local analysis phases.
