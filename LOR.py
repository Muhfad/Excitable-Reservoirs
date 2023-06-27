'''
Implementing lateral orthogonal rectification for identying periodic orbits in 
trained RNN. 
'''
# using simpson rule to calculate integrals
#%%
from scipy.integrate import simpson
import numpy as np
from numpy.linalg import norm, inv, det
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
    # assert np.all((t >= interval[0]) * (t <= interval[1]))
    coeff = F['results'] 
    n_freqs = F['n_freqs']
    N = F['dim']
    L = F['L'] + 2*1e-3
    try:
        n_points = len(t)
    except TypeError:
        n_points = 1
    if k > 0:
        # if we require a derivative, the constants vanishes 
        coeff[0][0] = coeff[0][1]
    C = np.matmul(np.array([1, 0.]), coeff[0]).reshape(-1, 1)
    output = C + np.zeros((N, n_points))
    for f in range(1, n_freqs):
        c = (2*np.pi*f/L)**k*np.cos(2*np.pi*f*t/L + k*np.pi/2).flatten()
        s = (2*np.pi*f/L)**k*np.sin(2*np.pi*f*t/L + k*np.pi/2).flatten()
        cs = np.c_[c, s].reshape(n_points, -1, 2)
        C_n = np.sum(coeff[f].T * cs, axis = 2).T
        output += C_n 
    return output



def Frenet_Frame(F, t):
    '''
    Takes a Fourier Approximation object F and return a Frenet Frame for it using
    Modified Gram-Schmidt Process.
    '''
    n_freqs = F['n_freqs']
    n = F['dim']
    k = n-1
    x_pred = np.zeros((n, n))
    for i in range(n):
        x_pred[i] = evaluate_fourier(F,t, i+1).T
    res = np.zeros((n, n))
    res[0] = x_pred[0]/norm(x_pred[0])
    for i in range(1, k):
        v = x_pred[i] - Projection(x_pred[i], res[:i])
        res[i] = v/norm(v)
        # res[i] = v
    
    # computing the last vector
    ix = np.arange(n)
    C = res[:k, :]
    for i in range(n):
        res[k][i] = (-1)**(n + i + 1)*det(C[:, ix != i])
    res[k] = res[k]/norm(res[k])
    return res, x_pred



def is_orthonormal(V, tol = 1e-6):
    '''
    checks if the rows of V are orthonormal to within
    tolerance level tol
    '''
    j = 0
    n = V.shape[0]
    for i in range(n):
        v = V[i]
        if np.abs(norm(v) - 1.0) > tol: 
            return False
        j = i + 1
        while j < n:
            u = V[j]
            if np.abs(np.inner(u, v)) > tol:
                return False
            j += 1
    return True



def orientation(V):
    '''
    checks if a an a set of basis vectors are positvely
    oriented. 
    '''
    res = 1
    if np.prod(np.diag(V)) < 0:
        res = -1
    return res



def Projection(v, U):
    ''' 
    projects v onto each of the rows of U and return their sum
    '''
    res = np.zeros_like(v)
    for i in range(U.shape[0]):
        u = U[i]
        res += np.inner(v, u)/np.inner(u, u)*u
    return res



def curvatures(Frame, n_diffs):
    '''
    calculates the curvatures of the Frenet frame Frame using the 
    corresponding (n x n) matrix of derivatives n_diffs 
    '''
    frame_diff = diff_Frame(n_diffs)
    k = n_diffs.shape[0] - 1
    res = np.zeros(k)
    for i in range(k):
        res[i] = np.inner(frame_diff[i], Frame[i+1])
    res /= norm(n_diffs[0])
    return res



def minors_of_curvatures(kappas):
    '''
    returns the principal minor of the matrix of curvatures
    '''
    n = len(kappas) + 1
    C = np.zeros((n, n))
    ix = np.arange(n)
    ix_m1 = (ix - 1) 
    cond1 = ix_m1 >= 0
    ix_p1 = (ix + 1) 
    cond2 = ix_p1 < n
    C[ix[cond1], ix_m1[cond1]] = -kappas
    C[ix[cond2], ix_p1[cond2]] = kappas
    # calculate minors
    M = det(C[1:, 1:])
    return M



def LOR_rhs(f, eta, ksi, gamma):
    '''
    Evaluates the LOR right hand side at eta and ksi using the Frenet 
    frame of the curve gamma

    ----------------------------------
    f: The right hand side of an ODE
    eta: reparameterisation of time in LOR
    ksi: reparameterisation of space in LOR
    gamma: An Fouriier object which parameterises a smooth curve. The 
    evaluate function can be used to evaluate gamma at each point of 
    eta
    '''
    Frame, n_diffs = Frenet_Frame(gamma, eta)
    gamma0 = evaluate_fourier(gamma, eta).T
    frame_diff = diff_Frame(n_diffs) 
    kappas = curvatures(Frame, n_diffs)
    M = minors_of_curvatures(kappas)
    k1 = kappas[0]
    Ngamma = Frenet[1:]
    Phi = gamma0 + np.sum(np.multiply(ksi, Ngamma.T), axis = 1)
    Tf = np.inner(f(Phi), Frame[0])
    Nf = np.sum(f(Phi) * Ngamma, axis = 1)

    norm_g1 = norm(n_diffs[0])
    osculator = (1 - ksi[0] * k1)
    eta_dot = Tf/(norm_g1 * osculator)
    # print(Nf.shape, Tf.shape, M, ksi.shape)
    ksi_dot = Nf - Tf * M * ksi/osculator

    return eta_dot, ksi_dot



def diff_unit(u, u_prime):
    '''
    differentiate the unit vector corresponding to u
    '''
    return u_prime/norm(u) - u*np.inner(u, u_prime)/norm(u)**3



def diff_inner(u, v, u_prime, v_prime):
    '''
    differentiate the inner product of u and v
    '''
    return np.inner(u, v_prime) + np.inner(v, u_prime)



def diff_Frame(n_diffs):
    '''
    returns the derivative of the tangents and normal vectors in the Frenet Frame
    excluding the last normal vector.
    -----------------------------------------
    n_diffs: An (n-1, n) matrix containing the first n-1 derivatives of the curve at eta. Here
    we anticipate that the first n derivative would have been calculated already
    in determing the Frenet Frame. 
    '''
    n = n_diffs.shape[1]
    # n+1 derivative at eta
    diffs = np.zeros((n-1, n))
    diffs[0] = diff_unit(n_diffs[0], n_diffs[1])
    for i in range(1, n-1):
        u = n_diffs[i + 1]
        u_curr = n_diffs[i]
        u_curr_prime = n_diffs[i+1]
        u_j = n_diffs[0]
        for j in range(i):
            u_next = n_diffs[j+1]
            j_inner_curr = np.inner(u_curr, u_j)
            inner_j = norm(u_j)**2
            u_1 = u_next * j_inner_curr / inner_j
            u_2 = - j_inner_curr/inner_j**2 * u_j * \
                 diff_inner(u_j, u_j, u_next, u_next)
            u_3 = u_j/inner_j * diff_inner(u_j, u_curr, u_next, u_curr_prime)
            u -= (u_1 + u_2 + u_3)
            u_j = u_next
        diffs[i] = u
    return diffs





#%%
t = np.linspace(-1, 1, 1000)
x1 = np.sin(t)**2
x1_prime = 3*t**2 - 10
x2 = np.sin(t)**2 - np.cos(t)
x2_prime = np.cos(t)**2
x3 = np.cos(t)
x4 = np.tan(t)
# x5 = t**2
X = np.c_[x1, x2, x3, x4]
X = X.T
  # %%
n_freqs = 10
gamma = FOURIER(X, t, n_freqs=n_freqs)
# n = 5
# k = n-1
x_pred = evaluate_fourier(F,t)
# for i in range(1, k):
#     x_pred = np.r_[x_pred, evaluate_fourier(F,t, k=i)]
#%%
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
