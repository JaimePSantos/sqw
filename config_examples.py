#!/usr/bin/env python3

"""
Configuration examples for static_cluster_logged.py

This file shows different configuration combinations for various use cases.
Copy the desired configuration to the top of static_cluster_logged.py
"""

# ============================================================================
# EXAMPLE 1: FULL PIPELINE (DEFAULT)
# Use this for complete experiments from scratch
# ============================================================================
FULL_PIPELINE = {
    'CALCULATE_SAMPLES_ONLY': False,
    'SKIP_SAMPLE_COMPUTATION': False,
    'ENABLE_PLOTTING': True,
    'CREATE_TAR_ARCHIVE': True,
    'RUN_IN_BACKGROUND': True
}

# ============================================================================
# EXAMPLE 2: SAMPLES ONLY (FOR CLUSTER COMPUTATION)
# Use this to compute only samples on a powerful cluster
# ============================================================================
SAMPLES_ONLY = {
    'CALCULATE_SAMPLES_ONLY': True,
    'SKIP_SAMPLE_COMPUTATION': False,
    'ENABLE_PLOTTING': False,
    'CREATE_TAR_ARCHIVE': False,
    'RUN_IN_BACKGROUND': True
}

# ============================================================================
# EXAMPLE 3: ANALYSIS ONLY (FOR LOCAL POST-PROCESSING)
# Use this to analyze existing samples locally with plotting
# ============================================================================
ANALYSIS_ONLY = {
    'CALCULATE_SAMPLES_ONLY': False,
    'SKIP_SAMPLE_COMPUTATION': True,
    'ENABLE_PLOTTING': True,
    'CREATE_TAR_ARCHIVE': True,
    'RUN_IN_BACKGROUND': False
}

# ============================================================================
# EXAMPLE 4: QUICK ANALYSIS (NO ARCHIVE)
# Use this for quick analysis without creating archives
# ============================================================================
QUICK_ANALYSIS = {
    'CALCULATE_SAMPLES_ONLY': False,
    'SKIP_SAMPLE_COMPUTATION': True,
    'ENABLE_PLOTTING': True,
    'CREATE_TAR_ARCHIVE': False,
    'RUN_IN_BACKGROUND': False
}

# ============================================================================
# EXAMPLE 5: HEADLESS COMPUTATION (NO PLOTTING)
# Use this for server environments without display
# ============================================================================
HEADLESS = {
    'CALCULATE_SAMPLES_ONLY': False,
    'SKIP_SAMPLE_COMPUTATION': False,
    'ENABLE_PLOTTING': False,
    'CREATE_TAR_ARCHIVE': True,
    'RUN_IN_BACKGROUND': True
}

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

def apply_config(config_name):
    """
    Apply a configuration to the script.
    
    Example usage:
    from config_examples import apply_config, SAMPLES_ONLY
    apply_config(SAMPLES_ONLY)
    """
    import static_cluster_logged
    
    for key, value in config_name.items():
        if hasattr(static_cluster_logged, key):
            setattr(static_cluster_logged, key, value)
            print(f"Set {key} = {value}")
        else:
            print(f"Warning: {key} not found in static_cluster_logged")

def print_config_summary():
    """Print a summary of all available configurations"""
    configs = {
        'FULL_PIPELINE': FULL_PIPELINE,
        'SAMPLES_ONLY': SAMPLES_ONLY,
        'ANALYSIS_ONLY': ANALYSIS_ONLY,
        'QUICK_ANALYSIS': QUICK_ANALYSIS,
        'HEADLESS': HEADLESS
    }
    
    print("Available Configurations:")
    print("=" * 50)
    
    for name, config in configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    print_config_summary()
