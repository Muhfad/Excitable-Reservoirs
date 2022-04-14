'''
Implementation of Echo State Networks from the paper:
    The “echo state” approach to analysing and training recurrent neural networks
     – with anErratum note

We assume that there are K input units, N internal units and L output units. We
may also have output feedback and connections from the input to the output and
from one output unit to another. Only the output weights are trainable. The
condition on the input weight matrix is that if the the principal singular value
is less than 1 and the spectral radius is more than 1, the ESP is guarenteed

W_in is (N x K)
W is (N x N)
W_back is (K x L)
'''

import numpy as np
from scipy import sparse # for creating sparse adjacency matrix


np.random.seed(200)

def internal_states(W_in, W, W_back, u, x_prev, y_prev, f = np.tanh):
    '''
    This function calculates the current state of the reservoir given the
    previous state x_prev, the current input sequence u and output feedback y_prev.
    '''
    return f(np.matmul(W_in, u) + np.matmul(W, x_prev) + np.matmul(W_back, y_prev))

def output(W_out, u, x, y_prev, f_out_inverse = np.arctanh):
    '''
    Calculates the approximate of the current output given the input u, the State
    of the reservoir x and the previous output y_prev
    '''
    return f_out_inverse(np.matmul(W_out, np.r_[u, x, y_prev]))


def Weights(U, N, Y, output_feedback, connectivity = 0.25):
    '''
    initialised the internal states (with N internal units), the input output feedback
    and output weight matrices
    '''
    L = Y.shape[0]
    if not output_feedback:
        W_back = np.zeros([N, L])
    else:
        W_back = np.random.rand([N, L])
    W = sparse.rand(N, N, density = connectivity).todense() # generate this so that the condition for ESP is satisfied
    print(W)
    W_in = np.random.rand(N, U.shape[0])
    return (W, W_in, W_back)


def ESN(U, Y, output_feedback, N):
    '''
    returns weights of the trained Echo state network
    '''
    W, W_in, W_back = Weights(U, N, Y, output_feedback = output_feedback)

    # the sizes of the different matrices
    K, n_obs = U.shape      # K is the length of an input sequence
    L, _ = Y.shape          # L is the number of output channels/units

    # randomly iniitialised internal states
    # X = (np.random.normal(scale = 5, size = N) / 10.0).reshape(N, 1)
    X = np.zeros([N, 1])

    # previous output history
    for i in range(n_obs - 1):
        x_n = internal_states(W_in, W, W_back,
                U[:, i].reshape(K, 1), X[:, i].reshape(N, 1), Y[:, i].reshape(L, 1))
        X = np.c_[X, x_n]

# m = Y.shape[1]
    # if output_feedback:
    #     Z = np.r_[U, X, Y[:, range(m-1)]].T
    # else:
    #     Z = np.r_[U, X, np.zeros(Y.shape)].T
    # Normal equations for multiple linear regressions
    # W_out = np.matmul(np.linal.inv(matmul(Z.T, Z)), Z.T, y)
    return X



# creating false dataset
L = 10 # length of input sequence
N = 100 # number of internal units
n_obs = 1000 # number of input sequences
U = 1000 * np.sin(np.arange(1, L+1) / 5)

for i in range(n_obs - 1):
    u = np.sin(np.arange(L + 1 + i, 2*L + 1 + i) / 5)
    U = np.c_[U, u] # this does columnwise concatenations

Y = 0.5 * U[[L-2, L - 1], :]
X = ESN(U, Y, output_feedback = False, N = N)
print(X)
X.shape


from matplotlib import pyplot as plt

plt.plot(np.array(range(1, n_obs + 1)), X.T[:, 2], '-', linewidth=2.5)
plt.plot(Y.T[12, :], 'o', linewidth=2.5)


plt.show()
