import numpy as np
import matplotlib.pyplot as plt


THETA = np.array([10,5])

VAR = np.array([1, 3])

D = np.array([1, -THETA[0]/THETA[1]])

def f(x,theta):
    return np.power(x,2)*theta[1]/2 + x * theta[0]

OPT = f(D[1], THETA)

xs = np.arange(-3,0, 0.001)
ys_0 = [f(x,THETA) for x in xs]

N = 100


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


theta_0 = np.random.normal(THETA[0], VAR[0], size = N)
theta_1 = np.random.normal(THETA[1], VAR[1], size = N)

n = minimizer[0]
theta_hat = np.zeros(2)
theta_hat[0] += np.sum(theta_0[:n])/n
theta_hat[1] += np.sum(theta_1[n:])/(N-n)
print(theta_hat)
x_hat1 = -theta_hat[0]/theta_hat[1]

ys_1 = [f(x,theta_hat) for x in xs]

n = 80
theta_hat = np.zeros(2)
theta_hat[0] += np.sum(theta_0[:n])/n
theta_hat[1] += np.sum(theta_1[n:])/(N-n)
print(theta_hat)
x_hat2 = -theta_hat[0]/theta_hat[1]

ys_2 = [f(x,theta_hat) for x in xs]

plt.axvline(x = D[1], color = 'green', linestyle = '--', label = 'True Optimum')
plt.axvline(x = x_hat1, color = 'orange', linestyle = '--', label = 'Decision from Optimal Allocation')
plt.axvline(x = x_hat2, color = 'purple', linestyle = '--', label = 'Decision from Suboptimal Allocation')

plt.plot(xs,ys_0, label = 'Truth', color = 'green')
plt.plot(xs,ys_1, label = 'Optimal allocation', color = 'orange')
plt.plot(xs,ys_2, label = 'Suboptimal allocation', color = 'purple')
plt.legend()
plt.title('Effect of Allocation on Decision')
#plt.show()
plt.savefig('effect.png', dpi=300)

