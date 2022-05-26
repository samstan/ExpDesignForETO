import numpy as np
import matplotlib.pyplot as plt

opt_means = np.array([2.09, 2.32, 2.30, 2.42, 2.93, 3.27, 4.44, 5.33, 9.27])
opt_stds = np.array([0.22, 0.23, 0.22, 0.22, 0.30, 0.29, 0.43, 0.56, 3.80])




uni_means = np.array([3.02, 2.75, 3.39, 3.59, 4.44, 5.14, 6.23, 8.47, 12.2])
uni_stds = np.array([0.25, 0.25, 0.32, 0.33, 0.39, 0.45, 0.54, 0.79, 1.19])

revs = np.array([0.567, 1, 1.557, 2.208, 2.926, 3.693, 4.497, 5.327, 6.179])

opt_means*=1e-3
opt_stds*=1e-3
uni_means*=1e-3
uni_stds*=1e-3

opt_means/=revs
opt_stds/=revs
uni_means/=revs
uni_stds/=revs

xs = np.arange(1,10)

plt.title(r'Normalized Regret vs. $-\theta_0$ when $\theta_1 = 1$')
plt.ylabel('Normalized Regret')
plt.xlabel(r'$-\theta_0$')
plt.errorbar(x = xs, y = opt_means, yerr = opt_stds, color = 'lightblue', label = 'Optimized Allocation', capsize = 5)
plt.errorbar(x = xs, y = uni_means, yerr = uni_stds, color = 'purple', label = 'Uniform Allocation', capsize = 5)
plt.legend()
plt.savefig('n_norm.png', dpi = 300)
# plt.show()

