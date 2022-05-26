import numpy as np

import matplotlib.pyplot as plt

THETA = np.array([-4,1])

x_star = 3.20794005
theta_2 = np.array([-4.32759732,  0.99833647])
x_hat2 = 3.15076535
theta_3 = np.array([-4.44910768,  1.15087082])
x_hat3 = 3.06157015

theta_2 = [-5.37986779,  1.29057743]
x_hat2 = 3.26423114 
theta_3 = [-4.67451368,  1.06261745]
x_hat3 = 3.46910882

# theta_2 = [-6.29154407,  1.53551186]
# x_hat2 = 3.20697273
# theta_3 = [-2.66870012,  0.65588744]
# x_hat3 = 3.59924642

def sig_min(theta,x):
    return x/(1+np.exp(theta[0]+theta[1]*x))

a = np.arange(3,4, 0.01)

y1 = [sig_min(THETA, e) for e in a]
y2 = [sig_min(theta_2, e) for e in a]
y3 = [sig_min(theta_3, e) for e in a]

plt.plot(a,y1)
plt.plot(a,y2)
plt.plot(a,y3)

plt.axvline(x = x_star, color = 'green', linestyle = '--', label = 'True Optimum')
plt.axvline(x = x_hat2, color = 'orange', linestyle = '--', label = 'Decision from Optimal Allocation')
plt.axvline(x = x_hat3, color = 'purple', linestyle = '--', label = 'Decision from Uniform Allocation')
plt.xlabel('Price')

plt.plot(a,y1, label = 'Truth', color = 'green')
plt.plot(a,y2, label = 'Optimal allocation', color = 'orange')
plt.plot(a,y3, label = 'Uniform allocation', color = 'purple')
plt.legend(loc = 'lower right',prop={'size': 9})
plt.title('Effect of Allocation on Decision')
# plt.show()
plt.savefig('effect_price.png', dpi=300)