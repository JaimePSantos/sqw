#!/usr/bin/env python3
"""
Angle noise experiment with both cluster deployment and crash-safe logging.
This version combines the cluster_deploy decorator with the crash_safe_log decorator
to provide both cluster optimization and comprehensive logging without extra code.
"""

import time
from cluster_module import cluster_deploy
from logging_module.crash_safe_logging import crash_safe_log

# Experiment parameters - shared between decorator and function
N = 10000
samples = 1
steps = N//4

@crash_safe_log(
    log_file_prefix="angle_experiment",
    heartbeat_interval=90.0,  # Longer intervals for cluster runs
    log_system_info=True
)
@cluster_deploy(
    experiment_name="angle_noise",
    noise_type="angle",
    N=N,
    samples=samples,
    venv_name="qw_venv",  # Custom environment name
    check_existing_env=True,           # Check for existing environment
    create_tar_archive=True,          # Disable TAR archiving for faster development
    use_compression=False              # No compression needed since no TAR
)
def run_angle_experiment():
    """Run the angle noise quantum walk experiment with cluster deployment and crash-safe logging."""
    
    # Import after cluster environment is set up
    import networkx as nx
    import numpy as np
    
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    from sqw.utils import random_angle_deviation
    
    # Import smart loading function
    from smart_loading import smart_load_or_create_experiment
    
    # Import shared functions from jaime_scripts  
    from jaime_scripts import prob_distributions2std

    print("Starting angle noise experiment with smart loading...")
    devs = [0, (np.pi/3)/2.5, (np.pi/3)*2]
    # Fixed parameters for all experiments
    base_angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}
    
    print(f"Parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Deviations: {[f'{dev:.3f}' for dev in devs]}")
    
    # Generate angles list for all deviations and samples
    angles_list_list = []
    for dev in devs:
        dev_samples = []
        for sample_idx in range(samples):
            if dev == 0:
                # No noise case - use perfect angles
                dev_samples.append(base_angles)
            else:
                # Generate random deviation for this sample
                sample_angles = random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps)
                dev_samples.append(sample_angles)
        angles_list_list.append(dev_samples)
    
    # Use smart loading - handles everything automatically!
    start_time = time.time()
    mean_results = smart_load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_or_angles_list=angles_list_list,
        tesselation_order_or_list=tesselation_order,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=devs,
        samples=samples,
        noise_type="angle",
        parameter_name="angle_dev",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )
    total_time = time.time() - start_time
    
    print(f"\n🎉 Smart loading completed in {total_time:.2f} seconds")
    
    # Calculate statistics for verification
    domain = np.arange(N) - N//2
    for i, dev_mean_prob_dists in enumerate(mean_results):
        if dev_mean_prob_dists and len(dev_mean_prob_dists) > 0:
            std_values = prob_distributions2std(dev_mean_prob_dists, domain)
            print(f"Dev {devs[i]:.3f}: Final std = {std_values[-1]:.3f}")

    print("Angle experiment completed successfully!")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    return {
        "devs": devs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time
    }


if __name__ == "__main__":
    run_angle_experiment()
