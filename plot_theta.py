import numpy as np
import matplotlib.pyplot as plt

means_opt = [0.502, 0.0448, 0.0169, 0.00242, 0.00136]
means_uni = [0.274, 0.0390, 0.0180, 0.00359, 0.00201]

xs = [10, 50, 100, 500, 1000]

plt.plot(xs,means_opt, label = 'Optimized')
plt.plot(xs,means_uni, label = 'Uniform')
plt.xlabel('Log number of samples')
plt.ylabel('Log regret')
plt.title(r'Regret vs. Number of Samples: $\theta$ = [-4,1]')
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.savefig('theta.png', dpi = 300)