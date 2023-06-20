'''
Author: Muhammed Fadera
Task: Generate the dataset for the n-bit flip-flop task. 

Task Description: Input u[k] for the n-bits flip-flop remains at 
(0, 0, ... {n times}) for a precified probability q and the 
remainig bit flips +/-1 at any of the positions happens with equal 
probability p/(n-1). 
'''
#%%
import numpy as np
from copy import deepcopy
from copy import copy
import random
from math import ceil
#%%
def sequence(N, q, n = 2, random_state = 42):
    '''
    n (int): is the number of memory bits
    N (int): is the time interval/length of the sequence
    q: probability that it stays at (0, 0, ..., {n times})


    Approach
    flip a coin, if it is 0, leave the input unchanged. If it is 1,
    choose between 1 to n on a uniform distribution and call the result
    k. Flip a coin again and choose between -1 and 1. 

    '''
    # delay = delay_times(N, mem_range)
    rng = np.random.RandomState(random_state)
    random.seed(random_state)
    inputs = np.zeros((N, n))
    output = np.ones((N, n))
    bit_changes = rng.binomial(1, p = q, size=N) == 1
    pos = np.where(bit_changes)[0]
    l = np.sum(bit_changes)
    location = np.random.randint(0, n, l)
    inputs[bit_changes, location] = np.random.choice([-1, 1], l)
    for i in range(l-1):
        for k in range(n):
            u = inputs[pos[i], k]
            if u != 0:
                output[pos[i]:pos[i+1], k] = u 
    output_label = np.apply_along_axis(convert_to_states, 1, output, n = n)
    return inputs.reshape(-1, n),  output.reshape(-1, n), output_label

def convert_to_states(x, n):
    '''
    given the number of bits and a vector of size n containing 1s and -1s, this 
    function converts -1 to 0 and returns the corresponding base 10 representation
    '''
    y = deepcopy(x)
    y[y == -1] = 0
    x = [2**i*y[i] for i in range(n-1, -1, -1)]
    return int(sum(x))

#%%
# def track_states(start_states, cum_times):
#     '''
#     '''
#     n = cum_times.shape[0]
#     curr_state = 1
#     start_states = deepcopy(start_states)
#     for i in range(1, n-1): 
#         if cum_times[i] != cum_times[i+1]: 
#             curr_state *= -1
#             start_states[cum_times[i]:cum_times[i+1]] = curr_state
#     return start_states

# def delay_times(N, mem_range = 4):
#     output = np.zeros(N, dtype=np.int8)
#     output[0] = 1
#     for i in range(1, N):
#         if i - mem_range >= 1:
#             output[i] = np.random.randint(i - mem_range, i+1)
#         else:
#             output[i] = np.random.randint(1, i+1)
#     return output



# %%