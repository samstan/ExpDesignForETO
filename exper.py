import numpy as np
import matplotlib.pyplot as plt

np.random.seed(3120)


THETA = np.array([10,5])

VAR = np.array([1, 3])

D = np.array([1, -THETA[0]/THETA[1]])


M = 300 #num replications
N = 1000 #num samples

def f(x,theta):
    return np.power(x,2)*theta[1]/2 + x * theta[0]

OPT = f(D[1], THETA)

error_naive = 0
error_ifo = 0

def showmin(a,n):
    currmin = np.inf
    minimizer = (0,0)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i+j == n:
                tmp = a[0]/i + a[1]/j
                if tmp<currmin:
                    currmin = tmp
                    minimizer = (i,j)
    #print(currmin)
    return minimizer


a = np.zeros(2)
for i in range(2):
    a[i] = np.power(D[i],2)*VAR[i]
minimizer = showmin(a,N)

error_naive = np.zeros(M)
error_ifo = np.zeros(M)

res = np.zeros((M,N-2))

for i in range(M):

    theta_0 = np.random.normal(THETA[0], VAR[0], size = N)
    theta_1 = np.random.normal(THETA[1], VAR[1], size = N)

    for n in range(1, N-1):
        theta_hat = np.zeros(2)
        theta_hat[0] += np.sum(theta_0[:n])/n
        theta_hat[1] += np.sum(theta_1[n:])/(N-n)
        x_hat = -theta_hat[0]/theta_hat[1]
        res[i,n-1] = f(x_hat, THETA) - OPT

trunc_l = 10
trunc = 100

regrets = res.mean(axis = 0)
stds = 1.96*res.std(axis = 0)/np.sqrt(M)

plt.errorbar(x = np.arange(1+trunc_l,N-1-trunc), y = regrets[trunc_l:-trunc], yerr = stds[trunc_l:-trunc], ecolor = 'lightblue', label = 'Regret')
plt.axvline(x = minimizer[0], color = 'red', label = 'Analytical Optimum')
plt.axvline(x = 500, color = 'purple', label = 'Naive Allocation')
plt.axvline(x = np.argmin(regrets)+1, color = 'darkorange', label = 'Empirical Optimum')
plt.legend()
plt.title('Quadratic Regret vs Sample Allocation for 1000 Samples')
plt.xlabel('Samples Allocated to Slope')
plt.ylabel('Regret over 300 Replications')
# plt.show()
plt.savefig('quadratic.png', dpi=300)