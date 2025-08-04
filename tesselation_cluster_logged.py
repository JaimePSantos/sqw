#!/usr/bin/env python3
"""
Tesselation noise experiment with both cluster deployment and crash-safe logging.
This version combines the cluster_deploy decorator with the crash_safe_log decorator
to provide both cluster optimization and comprehensive logging without extra code.
"""

import time
from cluster_module import cluster_deploy
from logging_module.crash_safe_logging import crash_safe_log


@crash_safe_log(
    log_file_prefix="tesselation_experiment",
    heartbeat_interval=30.0,  # Longer intervals for cluster runs
    log_system_info=True
)
@cluster_deploy(
    experiment_name="tesselation_noise",
    noise_type="tesselation_order",
    N=2000,
    samples=10,
    venv_name="qw_venv",              # Custom environment name
    check_existing_env=True,           # Check for existing environment
    create_tar_archive=False,           # Enable TAR archiving
    use_compression=False              # No compression
)
def run_tesselation_experiment():
    """Run the tesselation noise quantum walk experiment with cluster deployment and crash-safe logging."""
    
    # Import after cluster environment is set up
    import numpy as np
    import networkx as nx
    
    from sqw.tesselations import even_line_two_tesselation
    from sqw.states import uniform_initial_state
    from sqw.utils import tesselation_choice
    
    # Import smart loading function
    from smart_loading import smart_load_or_create_experiment
    
    # Import shared functions from jaime_scripts  
    from jaime_scripts import prob_distributions2std

    print("Starting tesselation noise experiment with smart loading...")
    
    # Parameters
    N = 3000
    steps = N//4
    samples = 1
    shift_probs = [0, 0.2, 0.5]  # Tesselation shift probabilities
    
    # Fixed parameters for all experiments
    base_angles = [[np.pi/3, np.pi/3]] * steps
    base_tesselation_order = [[0, 1]] * steps  # Perfect tesselation order
    initial_state_kwargs = {"nodes": [N//2]}
    
    print(f"Parameters: N={N}, steps={steps}, samples={samples}")
    print(f"Shift probabilities: {[f'{prob:.3f}' for prob in shift_probs]}")
    
    # Generate tesselation order list for all shift probabilities and samples
    tesselation_orders_list_list = []
    for shift_prob in shift_probs:
        prob_samples = []
        for sample_idx in range(samples):
            if shift_prob == 0:
                # No noise case - use perfect tesselation order
                prob_samples.append(base_tesselation_order)
            else:
                # Generate random tesselation order with shift probability
                sample_tesselation_order = tesselation_choice(
                    [[0, 1], [1, 0]], 
                    steps, 
                    [1 - shift_prob, shift_prob]
                )
                prob_samples.append(sample_tesselation_order)
        tesselation_orders_list_list.append(prob_samples)
    
    # Use smart loading - handles everything automatically!
    start_time = time.time()
    mean_results = smart_load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_or_angles_list=base_angles,
        tesselation_order_or_list=tesselation_orders_list_list,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=shift_probs,
        samples=samples,
        noise_type="tesselation_order",
        parameter_name="shift_prob",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )
    total_time = time.time() - start_time
    
    print(f"\n🎉 Smart loading completed in {total_time:.2f} seconds")
    
    # Calculate statistics for verification
    domain = np.arange(N) - N//2
    for i, prob_mean_prob_dists in enumerate(mean_results):
        if prob_mean_prob_dists and len(prob_mean_prob_dists) > 0:
            std_values = prob_distributions2std(prob_mean_prob_dists, domain)
            print(f"Shift prob {shift_probs[i]:.3f}: Final std = {std_values[-1]:.3f}")

    print("Tesselation experiment completed successfully!")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    return {
        "shift_probs": shift_probs,
        "N": N,
        "steps": steps,
        "samples": samples,
        "total_time": total_time
    }


if __name__ == "__main__":
    run_tesselation_experiment()
