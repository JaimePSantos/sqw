import matplotlib.pyplot as plt
import numpy as np

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