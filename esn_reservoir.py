'''
Author: Muhammed Fadera
Date: 12/04/2022
Aim: Set up a CTRNN in the excitable regime to solve a classification task. 

Dataset: MNIST: hand written digit recognition (filtered for 3 and 6). 

Preprocessing: MNIST contain 28 x 28 pixels images of hand written digits. Each image
will be 

    a. normalise to the range [0, 0.1] by dividing by 10 x 255.0 
    b. flatten into a numpy vector of size 28 x 28 = 784 and

feed to RNN in sequence. The last ouput of the RNN will be used for prediction and this will be optimised using simple least
squares regression.

Approach: 
    1. Create the adjacency matrix of a graph which a three layer block structure. 
       See graph_generator.py for more details. 
    2. Obtain the weight matrix between the internal states using 
       
       W = w_t 1 + (w_s - w_t) I + (w_p - w_t) A + (w_m - w_t)A^T
    
     where A is the adjacency matrix, 1 is the matrix of ones with the same dimension as A,
     I is the identity matrix of the same dimension as A and A^T is the transpose of A. The 
     parameters are chosen as follows 

     w_t = 0; w_s = 1; w_p = 0.3; w_m = -0.7

    3. The weight matrix between the input and the network is chosen from a uniform distribution 
       U(0, 1/3).


'''

#%%

from graph_generator import graph
import numpy as np

delta = 0.35

def params(delta = delta, theta = 0.5):

    epsilon = delta/8
    w_s = 1
    w_t = 0
    w_p = (theta - delta/2)
    w_m = -(w_s - theta) - delta/2
    return w_t, w_s, w_p, w_m, epsilon

def weight(A, n_input, delta=delta):
    '''
    Takes in the adjacency matrix A and the lenght n_input of the input vector
    and return the recurrent matrix and the input to network matrix 
    '''
    w_t, w_s, w_p, w_m, epsilon = params(delta=delta)
    # w_p = np.random.uniform(0.8*w_p, w_p, A.shape)
    n = A.shape[0]
    W_r = w_t * np.ones_like(A) + (w_s - w_t)*np.eye(n) + \
         (w_p - w_t) * A.T + (w_m - w_t) * A

    W_in = np.random.uniform(0, 1.3, size = (n, n_input))

    return W_r, W_in

# piecewise affine nonlinearity
def phi_piecewise(y, theta = 0.5, delta = delta):
    '''
    This is the piecewise affine function (3)
    '''
    epsilon = delta/8
    if y - theta < -2*epsilon:
        return 0
    elif y - theta > 2*epsilon:
        return 1
    return (y - theta)/(4*epsilon) + 0.5
# vectorize the fucntion to take its input as an array
phi_piecewise = np.vectorize(phi_piecewise)

def phi(y, theta = 0.5, delta = delta):
    '''
    This is the smooth nonlinearity (2)
    '''
    epsilon = delta/8
    return 1 / (1 +  np.exp(-1/epsilon * (y - theta)))

def propagate(init_cond, u, W_r, W_in, stepsize = 1):
    m, _ = u.shape
    n = W_r.shape[0]
    x_now = init_cond
    curr_reservoir_state = np.zeros((m, n))
    itenirary = np.zeros(m, dtype = np.int16)
    for j in range(m):
        x_n = (1 - stepsize)*x_now + np.matmul(phi_piecewise(x_now), W_r.T) + np.matmul(u[j, :], W_in.T)
        # finds the location of the equillibrium that the state is closest to
        curr_reservoir_state[j, :] = x_n
        # print(np.argmax(x_n))
        # print(p)
        x_now = x_n
    # print(itenirary)
    # return itenirary
    return curr_reservoir_state.flatten()

#%%
def reservoir(U, A, init_cond = None):
    '''
    propagates the discretize CTRNN using forward euler with step size = time constant = 1. 
    
    '''
    # this is to make sure that we get the same random initialisations
    np.random.seed(2022)


    N, m, k = U.shape

    W_r, W_in = weight(A, k)
    n = A.shape[0]
    m = W_in.shape[1]
    if init_cond is None:
            init_cond = np.random.uniform(size = n)

    # one hot vector encoding which equillibra was visited at time n
    # reservoir_itin = np.zeros((N, m))

    # contains the state of the reservoir at time n for each input
    reservoir_states = np.zeros((N, n*m))


    for i in range(N):
        u = U[i, :, :]
        # reservoir_itin[i, :] = propagate(init_cond, u, W_r, W_in)
        reservoir_states[i, :] = propagate(init_cond, u, W_r, W_in)
    cache = W_r, W_in, init_cond
    # return reservoir_itin, cache
    return reservoir_states, cache

def prediction(U, W_r, W_in, init_cond):

    N = U.shape[0]
    n = W_r.shape[0]
    m = W_in.shape[1]
    reservoir_states = np.zeros((N, n*m))
    # reservoir_itin = np.zeros((N, m))
    for i in range(N):
        u = U[i, :, :]
        # reservoir_itin[i, :] = propagate(init_cond, u, W_r, W_in)
        reservoir_states[i, :] = propagate(init_cond, u, W_r, W_in)

    # return reservoir_itin
    return reservoir_states


# %%

# %%


#------------------------------------------------------------------------
# Using itenararies instead of the entire state vector
def propagate_itin(init_cond, u, W_r, W_in, stepsize = 0.2):
    m, _ = u.shape
    n = W_r.shape[0]
    x_now = init_cond
    curr_reservoir_state = np.zeros((m, n))
    itenirary = np.zeros(m, dtype = np.int16)
    for j in range(m):
        x_n = (1 - stepsize)*x_now + np.matmul(phi_piecewise(x_now), W_r.T) + np.matmul(u[j, :], W_in.T)
        # finds the location of the equillibrium that the state is closest to
        itenirary[j] = np.argmax(x_n)
        # print(np.argmax(x_n))
        # print(p)
        x_now = x_n
    # print(itenirary)
    return itenirary


def reservoir_itin(U, A, init_cond = None):
    '''
    propagates the discretize CTRNN using forward euler with step size = time constant = 1. 
    
    '''
    # this is to make sure that we get the same random initialisations
    np.random.seed(2022)


    N, m, k = U.shape

    W_r, W_in = weight(A, k)
    n = A.shape[0]
    m = W_in.shape[1]
    if init_cond is None:
            init_cond = np.random.uniform(size = n)

    # one hot vector encoding which equillibra was visited at time n
    reservoir_itin = np.zeros((N, m))



    for i in range(N):
        u = U[i, :, :]
        reservoir_itin[i, :] = propagate_itin(init_cond, u, W_r, W_in)
    cache = W_r, W_in, init_cond
    return reservoir_itin, cache

def prediction_itin(U, W_r, W_in, init_cond):

    N = U.shape[0]
    n = W_r.shape[0]
    m = W_in.shape[1]
    reservoir_itin = np.zeros((N, m))
    for i in range(N):
        u = U[i, :, :]
        reservoir_itin[i, :] = propagate_itin(init_cond, u, W_r, W_in)

    return reservoir_itin
# %%
