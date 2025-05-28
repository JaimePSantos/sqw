from sqw.tesselations import even_cycle_two_tesselation,even_line_two_tesselation
from sqw.experiments_expanded import running,hamiltonian_builder,unitary_builder
from sqw.states import uniform_initial_state, amp2prob
from sqw.statistics import states2mean, states2std, states2ipr, states2survival
from sqw.plots import final_distribution_plot, mean_plot, std_plot, ipr_plot, survival_plot
from sqw.utils import random_tesselation_order, random_angle_deviation, tesselation_choice

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

N = 300
# T = even_cycle_two_tesselation(N)
T = even_line_two_tesselation(N)
G = nx.cycle_graph(N)

steps = N
angles = [[np.pi/3, np.pi/3]] * steps
tesselation_order = [[0,1] for x in range(steps)]
initial_state = uniform_initial_state(N, nodes = [N//2])

angles_noisy = random_angle_deviation([np.pi/3, np.pi/3], [.2, .2], steps)

final_states_no_noise = running(G, T, steps, 
                 initial_state, 
                 angles = angles, 
                 tesselation_order = tesselation_order)

final_states_noise = running(G, T, steps, 
                 initial_state, 
                 angles = angles_noisy, 
                 tesselation_order = tesselation_order)

# Calculate domain (node indices)
domain = np.arange(N)

# Calculate standard deviation for both cases
std_no_noise = states2std(final_states_no_noise, domain)
std_noise = states2std(final_states_noise, domain)

# # Calculate survival probability for both cases (using initial node)
initial_node = N//2
survival_no_noise = states2survival(final_states_no_noise, initial_node)
survival_noise = states2survival(final_states_noise, initial_node)

# Plot standard deviation vs time
plt.figure(figsize=(8, 5))
plt.plot(std_no_noise, label='No Noise')
plt.plot(std_noise, label='With Noise')
plt.xlabel('Time Step')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation vs Time')
plt.legend()
plt.tight_layout()
plt.show()

# Plot survival probability vs time
plt.figure(figsize=(8, 5))
plt.plot(survival_no_noise, label='No Noise')
plt.plot(survival_noise, label='With Noise')
plt.xlabel('Time Step')
plt.ylabel('Survival Probability')
plt.title('Survival Probability vs Time')
plt.legend()
plt.tight_layout()
plt.show()

# Plot probability distribution after 80 steps for the noisy case
prob_dist_noise = np.abs(final_states_noise[-1])**2
plt.figure(figsize=(8, 5))
plt.plot(domain, prob_dist_noise, label='Noisy Case')
plt.xlabel('Node')
plt.ylabel('Probability')
plt.title('Probability Distribution after 80 Steps (Noisy)')
plt.legend()
plt.tight_layout()
plt.show()
