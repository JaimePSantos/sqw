from sqw.tesselations import even_cycle_two_tesselation,even_line_two_tesselation
from sqw.states import uniform_initial_state
from sqw.utils import tesselation_choice

from jaime_scripts import (
    plot_std_vs_time_qwak,
    plot_single_timestep_qwak,
    prob_distributions2std,
    smart_load_or_create_experiment  # New intelligent loading function
)

import networkx as nx
import numpy as np

if __name__ == "__main__":
    N = 100
    steps = N//4
    samples = 10  # Number of samples per shift probability
    angles = [[np.pi/3, np.pi/3]] * steps  # Fixed angles, no noise
    initial_state_kwargs = {"nodes": [N//2]}

    # List of shift probabilities for tesselation order switching
    shift_probs = [0, 0.2, 0.5]
    
    # Generate tesselation orders for each shift probability and sample
    tesselation_orders_list = []  # [shift_prob][sample] -> tesselation_order
    for shift_prob in shift_probs:
        shift_tesselation_orders = []
        for sample_idx in range(samples):
            if shift_prob == 0:
                # No noise case - use fixed tesselation order
                shift_tesselation_orders.append([[0, 1]] * steps)
            else:
                # Noisy tesselation order - generate different random sequence for each sample
                tesselation_order = tesselation_choice([[0, 1], [1, 0]], steps, [1 - shift_prob, shift_prob])
                shift_tesselation_orders.append(tesselation_order)
        tesselation_orders_list.append(shift_tesselation_orders)

    print(f"Running experiment for {len(shift_probs)} different tesselation shift probabilities with {samples} samples each...")
    print(f"Shift probabilities: {shift_probs}")

    # Use the new smart loading function instead of the old approach
    print("Using smart loading (probabilities → samples → create)...")
    results_list = smart_load_or_create_experiment(
        graph_func=nx.cycle_graph,
        tesselation_func=even_line_two_tesselation,
        N=N,
        steps=steps,
        angles_or_angles_list=angles,
        tesselation_order_or_list=tesselation_orders_list,
        initial_state_func=uniform_initial_state,
        initial_state_kwargs=initial_state_kwargs,
        parameter_list=shift_probs,
        samples=samples,
        noise_type="tesselation_order",
        parameter_name="prob",
        samples_base_dir="experiments_data_samples",
        probdist_base_dir="experiments_data_samples_probDist"
    )

    print(f"Got results for {len(results_list)} tesselation experiments")

    # Calculate statistics for plotting using mean results (probability distributions)
    domain = np.arange(N) - N//2  # Centered domain for std calculation
    stds = []
    for i, mean_prob_dists in enumerate(results_list):
        if mean_prob_dists and len(mean_prob_dists) > 0 and all(state is not None for state in mean_prob_dists):
            # These should be probability distributions from smart loading
            std_values = prob_distributions2std(mean_prob_dists, domain)
            stds.append(std_values)
            print(f"Tesselation {i} (shift_prob={shift_probs[i]:.3f}): {len(std_values)} std values")
        else:
            print(f"Tesselation {i} (shift_prob={shift_probs[i]:.3f}): No valid mean probability distributions")
            stds.append([])

    # Plot standard deviation vs time for all tesselation shifts
    print("\nPlotting standard deviation vs time...")
    plot_std_vs_time_qwak(stds, shift_probs, title_prefix="Tesselation shift (mean)", parameter_name="prob")

    # Plot probability distributions at the final timestep (last time step)
    final_timestep = steps - 1  # Last timestep (0-indexed)
    print(f"\nPlotting final probability distributions at timestep {final_timestep}")
    
    # Plot final distributions for all shift probabilities
    plot_single_timestep_qwak(results_list, shift_probs, final_timestep, domain, "Tesselation shift (final step)", "prob")
