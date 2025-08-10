# How to Use static_noise_clean_refactored.py

## Method 1: Edit DEFAULT_CONFIG directly in the code

Open `static_noise_clean_refactored.py` and find the `DEFAULT_CONFIG` section at the top:

```python
DEFAULT_CONFIG = {
    'N': 50,                    # Change this to your desired system size
    'theta': np.pi/4,           # Change this to your desired theta
    'steps': None,              # Change this to your desired steps (None = auto)
    'samples': 10,              # Change this to your desired sample count
    'init_nodes': None,         # Change this to your desired initial nodes
    'deviation_ranges': [       # Change these ranges as needed
        (0.0, 0.0),            # No noise
        (0.0, 0.1),            # Small positive noise
        (0.0, 0.2),            # Larger positive noise
    ],
    'run_analysis': True,       # Set to False to skip analysis
    'save_results': True        # Set to False to skip saving
}
```

Then simply run:
```bash
python static_noise_clean_refactored.py
```

## Method 2: Use the provided presets

Uncomment one of the preset configurations:

```python
# For production runs:
DEFAULT_CONFIG = {
    'N': 100, 'theta': np.pi/3, 'steps': 20, 'samples': 50,
    'deviation_ranges': [(0.0, 0.0), (0.0, 0.05), (0.0, 0.1), (0.0, 0.15), (0.0, 0.2)],
    'run_analysis': True, 'save_results': True
}

# For quick testing:
DEFAULT_CONFIG = {
    'N': 20, 'theta': np.pi/4, 'steps': 5, 'samples': 3,
    'deviation_ranges': [(0.0, 0.0), (-0.1, 0.1), (0.0, 0.2)],
    'run_analysis': False, 'save_results': False
}
```

## Method 3: Use the quick_config helper

```python
# Modify just the parameters you want to change:
DEFAULT_CONFIG = quick_config(N=100, samples=20, run_analysis=False)

# Or change multiple parameters:
DEFAULT_CONFIG = quick_config(
    N=200, 
    samples=100, 
    deviation_ranges=[(0.0, 0.0), (0.0, 0.05), (0.0, 0.1), (0.0, 0.15), (0.0, 0.2), (0.0, 0.3)]
)
```

## Examples of Different Deviation Ranges

```python
# No noise comparison
'deviation_ranges': [(0.0, 0.0)]

# Only positive noise
'deviation_ranges': [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3)]

# Symmetric noise around zero
'deviation_ranges': [(0.0, 0.0), (-0.1, 0.1), (-0.2, 0.2)]

# Asymmetric noise ranges
'deviation_ranges': [(0.0, 0.0), (0.0, 0.1), (-0.1, 0.05), (-0.2, -0.1)]

# Mixed positive and negative
'deviation_ranges': [(0.0, 0.0), (0.05, 0.15), (-0.15, -0.05), (-0.1, 0.2)]
```

## Quick Examples to Copy-Paste

### Small test run:
```python
DEFAULT_CONFIG = {
    'N': 20, 'theta': np.pi/4, 'steps': 5, 'samples': 3,
    'deviation_ranges': [(0.0, 0.0), (0.0, 0.1)],
    'run_analysis': False, 'save_results': False
}
```

### Medium production run:
```python
DEFAULT_CONFIG = {
    'N': 100, 'theta': np.pi/4, 'steps': 25, 'samples': 20,
    'deviation_ranges': [(0.0, 0.0), (0.0, 0.05), (0.0, 0.1), (0.0, 0.15), (0.0, 0.2)],
    'run_analysis': True, 'save_results': True
}
```

### Large production run:
```python
DEFAULT_CONFIG = {
    'N': 200, 'theta': np.pi/3, 'steps': 50, 'samples': 100,
    'deviation_ranges': [(0.0, 0.0), (0.0, 0.02), (0.0, 0.05), (0.0, 0.1), (0.0, 0.15), (0.0, 0.2), (0.0, 0.3)],
    'run_analysis': True, 'save_results': True
}
```

Just edit the file, save it, and run `python static_noise_clean_refactored.py` - no command line arguments needed!
