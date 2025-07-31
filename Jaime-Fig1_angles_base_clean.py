from sqw.tesselations import even_cycle_two_tesselation
from sqw.experiments import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the crash-safe logging decorator from the logging module
from logging_module import crash_safe_log

@crash_safe_log(log_file_prefix="sqw_execution_base", heartbeat_interval=10.0)
def main_computation():
    """
    Main computation with crash-safe logging via decorator from logging module.
    This replaces the original complex logging implementation with a simple decorator.
    """
    N = 200
    T = even_cycle_two_tesselation(N)
    G = nx.cycle_graph(N)

    steps = N
    angles = [[np.pi/3, np.pi/3]] * steps
    tesselation_order = [[0,1] for x in range(steps)]
    initial_state = uniform_initial_state(N, nodes = [N//2, N//2+1])

    states_ua_ut = running(G, T, steps, 
                     initial_state, 
                     angles = angles, 
                     tesselation_order = tesselation_order)

    final_dist = [amp2prob(x) for x in states_ua_ut]

    plt.plot(final_dist[30], label='Initial state')
    plt.show()
    
    return final_dist

if __name__ == "__main__":
    # Simply call the decorated function - all logging is handled automatically by the module
    result = main_computation()
    print(f"Computation completed successfully. Result length: {len(result) if result else 'None'}")
