import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

def final_distribution_plot(prob, domain):
    fig, ax = plt.subplots(dpi = 50, figsize = (10,5))

    ax.plot(domain, prob, linewidth = 3)
    ax.set_xlabel(r'$x$',fontsize = 20)
    ax.set_ylabel(r'$P(x)$',fontsize = 20)
    ax.tick_params(axis = 'x', labelsize = 15)
    ax.tick_params(axis = 'y', labelsize = 15)
    plt.show()


    
def mean_plot(mean_values, steps):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.plot(range(steps + 1), mean_values,linewidth = 3)
    plt.xlabel(r'$t$',fontsize = 20)
    plt.ylabel(r'$\mu(t)$', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(range(0, steps + 1), fontsize = 15, rotation = 45)
    plt.show()
    
def std_plot(std_values, steps):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.plot(range(steps + 1), std_values,linewidth = 3)
    plt.xlabel(r'$t$',fontsize = 20)
    plt.ylabel(r'$\sigma(t)$', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(range(0, steps + 1), fontsize = 15, rotation = 45)
    plt.show()
    
def ipr_plot(ipr_values, steps):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.plot(range(steps + 1), ipr_values,linewidth = 3)
    plt.xlabel(r'$t$',fontsize = 20)
    plt.ylabel(r'$IPR(t)$', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(range(0, steps + 1), fontsize = 15, rotation = 45)
    plt.show()
    
def survival_plot(survival_values, steps):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.loglog(range(steps + 1), survival_values,linewidth = 3)
    plt.xlabel(r'$t$',fontsize = 20)
    plt.ylabel(r'$S(t)$', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15, rotation = 45)
    plt.ylim([np.min(survival_values), np.max(survival_values)])
    plt.show()

def tesselation_plot(T):
    H = nx.Graph()
    
    colors = ['#C75656', '#51ACB8','#63A360','#F0E87A']
    
    for t in range(len(T)):
        for u,v in T[t]:
            H.add_edge(u,v, color = colors[t])
    
    edges = H.edges()
    colors = [H[u][v]['color'] for u,v in edges]
    
    plt.figure(dpi = 50)
    nx.draw_kamada_kawai(H, edge_color=colors, width = 5, node_color = '#8E9699')
    plt.show()

def square_grid_plot(prob, N):
    x = np.arange(0, N, 1)
    y = np.arange(0, N, 1)
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots()

    final_dist = np.zeros((N,N))
    for y in range(N):
        for x in range(N):
            final_dist[y,x] = prob[x + y*N]
    
    im = ax.imshow(final_dist,interpolation='None',cmap='viridis')
    
    ratio = 0.5
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    # ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
    ax.set_xlabel('x',fontsize = 14)
    ax.set_ylabel('y',fontsize = 14)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    ax.set_xlim([0, N-1])
    
    ax.set_yticks(np.arange(0, N, 2))
    ax.set_xticks(np.arange(0, N, 2))
    
    cbar = fig.colorbar(im,fraction=0.035, pad=0.04)
    cbar.set_label('Probability', rotation=270,labelpad=20,fontsize = 14)