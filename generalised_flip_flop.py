#%%
from unittest import result
import matplotlib.pyplot as plt
from matplotlib import style
from stochastic.processes.diffusion import VasicekProcess
import numpy as np
from copy import deepcopy

style.use('ggplot')


#%%
cox = VasicekProcess( vol = 0.3, mean = 1.3, t = 10)
cox1 = VasicekProcess( vol = 0.3, mean = 1.3, t = 10)

np.random.seed(2020)

X = (cox.sample(1000, initial=1.3)) + 0.5
np.random.seed(2021)
Y = (cox1.sample(1000, initial=1.4)) + 0.5
t = (cox1.times(1000))


switch = np.ones_like(Y)
n = len(switch)
k = n//10
switch[k:3*k] = 0

switch[6*k:8*k] = 0


result = deepcopy(X)
result[switch == 0] = Y[switch == 0]

result = result + np.abs(np.random.normal(scale=0.01, size=n))

plt.plot(t, X, 'b', label = 'input 1')
plt.plot(t, Y, 'g', label = 'input 2')
plt.plot(t, result, 'r', label = 'output')
plt.plot(t, switch, 'k', label = 'switch')
plt.legend()
    # plt.ylim((-2, 5))
plt.show()
# %%
