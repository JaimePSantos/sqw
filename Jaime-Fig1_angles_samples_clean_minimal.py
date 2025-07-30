import networkx as nx
import numpy as np

from sqw.tesselations import even_line_two_tesselation
from sqw.states import uniform_initial_state
from sqw.utils import random_angle_deviation

    
    
    

if __name__ == "__main__":
    N = 2000
    steps = N//4
    samples = 10  # Number of samples per deviation
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state_kwargs = {"nodes": [N//2]}

    devs = [0, (np.pi/3)/2.5,(np.pi/3) * 2]
    angles_list_list = []
    
    for dev in devs:
        dev_angles_list = []
        for sample_idx in range(samples):
            if dev == 0:
                # No noise case - use perfect angles
                dev_angles_list.append([[np.pi/3, np.pi/3]] * steps)
            else:
                dev_angles_list.append(random_angle_deviation([np.pi/3, np.pi/3], [dev, dev], steps))
        angles_list_list.append(dev_angles_list)

    print(f"Running experiment for {len(devs)} different angle noise deviations with {samples} samples each...")
    print(f"Angle devs: {devs}")

    # Use the new smart loading function instead of the old approach
    print("Using smart loading (probabilities → samples → create)...")
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
