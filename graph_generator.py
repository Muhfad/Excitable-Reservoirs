#%%
import numpy as np



#%%
def graph(m, n, k):
    '''
    Generate the adjacency matrix of a directed graph with N = m + n + k
    vertices. The resulting matrix is blocked with the first m vertices
    connects only to the nodes at n-1, ..., m-1. The block at index n+1,
    ..., m connects only to the block at m+1, ..., k and the block at 
    m+1, ..., k connects to the block at 1, ..., m 
    '''
    N = m + n + k
    A = np.zeros((N, N))
    first, second, third, end = 0, m, m+n, N
    A[first:second, second:third] = 1
    A[second:third, third:end] = 1
    A[third:end, first:second] = 1
    
    return A

#%%
def sparsity(A):
    '''
    calculates the percentage of non-zero entries of A
    '''
    m, n = A.shape
    size = m * n
    return np.sum(A) / size
#%%
A = graph(1, 2, 1)

sparsity(A)
# %%
