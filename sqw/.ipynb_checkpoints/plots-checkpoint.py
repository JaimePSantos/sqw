import matplotlib.pyplot as plt

def final_distribution_plot(prob, domain):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.plot(domain, prob, linewidth = 3)
    plt.xlabel(r'$x$',fontsize = 20)
    plt.ylabel(r'$P(x)$',fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15, rotation = 45)
    plt.show()
    
def mean_values_plot(mean_values, steps):
    plt.figure(dpi = 50, figsize = (10,5))
    plt.plot(range(steps + 1), mean_values,linewidth = 3)
    plt.xlabel(r'$x$',fontsize = 20)
    plt.ylabel(r'$\mu(x)$', fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15, rotation = 45)
    plt.show()
    