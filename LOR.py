'''
Implementing lateral orthogonal rectification for identying periodic orbits in 
trained RNN. 
'''
# using simpson rule to calculate integrals
#%%
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

#%%
def FOURIER(x, tspan, n_freqs):
    '''
    calculates the coefficients of a fourier approximation of x on tspan upto 
    n_freqs frequencies. We assume that f(tspan) = x
    '''
    assert n_freqs >= 1
    L = np.abs(tspan[-1] - tspan[0])
    x_dim = x.shape
    if len(x_dim) == 1:
        n = 1
    else:
        n = x_dim[0]
    results = np.zeros((n_freqs+1, 2, n))
    A_0 = 1/L * simpson(x, tspan).reshape(1, -1)
    B_0 = np.zeros((1, n))
    results[0] = np.r_[A_0, B_0]
    for f in range(1, n_freqs+1):
        A_n = 2/L * simpson(x*np.cos(2*np.pi*f*tspan/L), tspan) 
        B_n =  2/L * simpson(x*np.sin(2*np.pi*f*tspan/L), tspan) 
        results[f] = np.r_[A_n.reshape(1, -1), B_n.reshape(1, -1)]
    return {'results':results,
            'n_freqs': n_freqs, 'L': L, 
            'interval':(tspan[0], tspan[-1]), 
            'dim':n, 'n_points':len(tspan)}


def evaluate_fourier(F, t, k = 0):
    '''
    evaluates the k^th derivative of a FOURIER object for every points in t. 
    Every element of t must be in the interval of F. k => 0
    '''
    t = np.array(t)
    interval = F['interval']
    assert np.all((t >= interval[0]) * (t <= interval[1]))
    coeff = F['results'] 
    n_freqs = F['n_freqs']
    N = F['dim']
    L = F['L'] + 2*1e-3
    t[0] = t[0] - 1e-3
    t[-1] = t[0] + 1e-3
    n_points = len(t)
    if k > 0:
        # if we require a derivative, the constants vanishes 
        coeff[0][0] = coeff[0][1]
    C = np.matmul(np.array([1, 0.]), coeff[0]).reshape(-1, 1)
    output = C + np.zeros_like(x)
    for f in range(1, n_freqs):
        c = (2*np.pi*f/L)**k*np.cos(2*np.pi*f*t/L + k*np.pi/2).flatten()
        s = (2*np.pi*f/L)**k*np.sin(2*np.pi*f*t/L + k*np.pi/2).flatten()
        cs = np.c_[c, s].reshape(n_points, -1, 2)
        C_n = np.sum(coeff[f].T * cs, axis = 2).T
        output += C_n 
    return output

# def Frenet_Frame(F, x):
#     '''
#     Takes a Fourier Approximation and return a Frenet Frame for it
#     '''
#     resuls = dict()
#     n = F['dim']
#     for i in range(1, n+1):
    
#%%
t = np.linspace(-5, 5, 1000)
x = t**3 - 10*t + 10 
x_prime = 3*t**2 - 10
y = np.sin(t)
y_prime = np.cos(t) 
X = np.c_[x, y]
X = X.T
plt.plot(t, x)
plt.show()
  # %%
n_freqs = 10
F = FOURIER(X, t, n_freqs=n_freqs)
x_pred = evaluate_fourier(F,t, k=1)
#%%
plt.subplot(2, 1, 1)
# plt.plot(t, X[0], label = 'actual')
plt.plot(t, x_prime, label = 'actual')
plt.plot(t, x_pred[0], label = f'approximated $f={n_freqs}$')
plt.legend()
plt.show()
plt.subplot(2, 2, 1)
# plt.plot(t, X[1], label = 'actual')
plt.plot(t, y_prime, label = 'actual')
plt.plot(t, x_pred[1], label = f'approximated $f={n_freqs}$')
plt.legend()
plt.show()
# %%
