import numpy as np
import matplotlib.pyplot as plt

i = 4

# minizers = np.load('mins.npy')

THETA = np.array([-i,1])

def sig_min(theta,x):
    return 1/(1+np.exp(theta[0]+x))

def func(x):
    return x*sig_min(THETA,x)

a = np.arange(0,10, 0.1)

y = [func(e) for e in a]

# minizer = minizers[i-1]

minizer = np.array([ 3.18,  3.61,  3.91,  7.19, 28.58, 31.25, 10.5 ,  5.23,  3.24,
        3.31])

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_ylabel('Proportion', color=color)  # we already handled the x-label with ax1
ax1.bar(np.arange(10), minizer/np.sum(minizer), color = color, label = 'Optimal Allocation')


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:purple'

ax2.set_ylabel('Function value', color=color)
ax2.plot(a, y, color='purple', label = 'Revenue Function')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Revenue Function and Optimal Allocation: Theta = [-4,1]')
ax1.set_xlabel('Price')

plt.savefig('neg4.png',dpi = 300)
# 

# plt.title('Revenue Function and Optimal Allocation: Theta = [-4,1]')
# plt.plot(a,y,color = 'orange', label = 'Revenue Function')
# plt.bar(np.arange(10)+0.5, minizer/np.sum(minizer),color = 'royalblue', label = 'Optimal Allocation')
# plt.legend()
# plt.savefig('neg4.png',dpi = 300)

