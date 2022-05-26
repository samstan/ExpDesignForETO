import numpy as np
import itertools
import time
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from tqdm import tqdm
N = 100 #number of plug in samples

N_ = 500 #number of samples to allocate

def sig_min(theta,x):
    return 1/(1+np.exp(theta@x))

def rev(theta, x):
    tmp = np.concatenate([[1],x])
    return -x[0]*sig_min(theta,tmp)

def getX(theta):
    def obj(x):
        return rev(theta,x)
    return np.concatenate([[1],minimize(obj,[1]).x]) 

def partitions(n, k):
    '''credit https://stackoverflow.com/questions/28965734/general-bars-and-stars'''
    for c in itertools.combinations(range(n+k-1), k-1):
        yield [b-a-1 for a, b in zip((-1,)+c, c+(n+k-1,))]


def getXfromAlloc(alloc):
    tmp = np.sum(alloc)
    X = []
    for i in range(10):
        X+=[i]*alloc[i]
    X = np.array(X)
    X = np.reshape(X,(-1,1))
    X_ = np.ones((tmp,1))
    X = np.concatenate([X_,X], axis = 1) 
    return X

def getAllocfromBars(b):
    b_sort = np.sort(b)
    diffs = np.ediff1d(b_sort)-1
    return np.concatenate((np.array([b_sort[0]]),diffs,np.array([N_ + 8 - b_sort[-1]])))

def getYfromX(X, theta):
    ''' always under true model THETA '''
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        if np.random.uniform()<sig_min(theta,X[i]):
            y[i] = 1
    return y

def filterY(y_shared, alloc):
    n = np.sum(alloc)
    dividers = N_
    res = []
    for i in range(len(alloc)):
        for j in range(alloc[i]):
            res.append(y_shared[i*dividers+j])
    return np.array(res)


def sim(theta, M):
    def rev_actual(x):
        return x[1]*sig_min(theta,x)
    x_star = getX(theta)
    regrets_smart = np.zeros(M)
    regrets_naive = np.zeros(M)

    min_average = np.zeros(10)
    for m in tqdm(range(M)): #to show progress bar
    # for m in range(M):
        X = np.zeros(N)
        for i in range(10):
            for j in range(10):
                X[i*10 + j] = i

        X = np.reshape(X,(-1,1))

        X_ = np.ones((N,1))

        X = np.concatenate([X_,X], axis = 1)

        y = getYfromX(X, theta)

        clf = LogisticRegression(penalty = 'none', fit_intercept = False).fit(X, 1-y)
        theta_0 = clf.coef_[0]

        x_hat = getX(theta_0)

        def func(x):
            n = x.shape[0]
            W = np.zeros(n)
            for i in range(n):
                W[i] = sig_min(theta_0,x[i])*(1-sig_min(theta_0,x[i]))
            var =  np.linalg.inv(x.T@np.diag(W)@x)
            d = np.zeros(2)
            c = sig_min(theta_0,x_hat)
            d[0] = c*(1-c)+theta_0[1]*x_hat[1]*c*(1-c)*(2*c-1)
            d[1] = 2*x_hat[1]*c*(1-c)+np.power(x_hat[1],2)*theta_0[1]*c*(1-c)*(2*c-1)
            return np.dot(d.T ,np.dot(var,d))

        mini = np.inf
        minizer = np.zeros(10)
        for _ in range(1000):
            bars = np.random.choice(np.arange(N_+9),size = 9, replace = False)
            alloc = getAllocfromBars(bars)
            X_l = getXfromAlloc(alloc)
            try: 
                tmp = func(X_l)
                if tmp<0: #numerical stability issues
                    pass
                else:
                    if tmp<mini:
                        mini = tmp
                        minizer = alloc
            except: #numerical stability issues
                pass
        min_average += minizer/M

        y_shared = getYfromX(getXfromAlloc(np.array([N_]*10)),theta)
        y_2 = filterY(y_shared, minizer)
        y_3 = filterY(y_shared, np.array([N_//10]*10))

        if np.min(y_2)==np.max(y_2):#if all equal, skip, otherwise error in log reg
            regrets_smart[m] = rev_actual(x_star) #mark to skip
        else:
            X_2 = getXfromAlloc(minizer)
            clf = LogisticRegression(penalty = 'none', fit_intercept = False).fit(X_2, 1-y_2)
            theta_2 = clf.coef_[0]
            x_hat2 = getX(theta_2)
            regrets_smart[m] = rev_actual(x_star) - rev_actual(x_hat2)
        if np.min(y_3)==np.max(y_3):
            regrets_naive[m] = rev_actual(x_star) #'' 
        else: 
            X_3 = getXfromAlloc(np.array([N_//10]*10))
            clf = LogisticRegression(penalty = 'none', fit_intercept = False).fit(X_3, 1-y_3)
            theta_3 = clf.coef_[0]
            x_hat3 = getX(theta_3)
            regrets_naive[m] = rev_actual(x_star) - rev_actual(x_hat3)   



    regrets_smart = np.delete(regrets_smart, regrets_smart==-1)
    regrets_naive = np.delete(regrets_naive, regrets_naive==-1)

    return min_average, np.mean(regrets_smart), np.std(regrets_smart), np.mean(regrets_naive), np.std(regrets_naive)

THETAS = np.array([[-1,1],[-2,1],[-3,1],[-4,1],[-5,1],[-6,1],[-7,1],[-8,1],[-9,1]])
# THETAS = np.array([[-4,1]])
len_theta = len(THETAS)
minizers = np.zeros((len_theta, 10))

M = 1000
for i,theta in enumerate(THETAS):
    print('theta: ', theta)
    a,b,c,d,e = sim(theta, M)
    minizers[i] = a
    print('smart', b,1.96*c/np.sqrt(M))
    print('naive', d,1.96*e/np.sqrt(M))
