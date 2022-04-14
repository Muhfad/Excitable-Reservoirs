#%%
import numpy as np
import tensorflow as tf

from esn_reservoir import *

#%%
(u_train, y_train), (u_test, y_test) = tf.keras.datasets.mnist.load_data()


#%%
def filter_36(u, y):
    keep = (y == 8) | (y == 3)
    u, y = u[keep], y[keep]
    y = y == 3
    return u, y


# %%
u_train, y_train = filter_36(u_train, y_train)
u_test, y_test = filter_36(u_test, y_test)

print("Number of filtered training euamples:", len(u_train))
print("Number of filtered test euamples:", len(u_test))
# %%
u_train, u_test = u_train[..., np.newaxis]/(255.0), u_test[..., np.newaxis]/(255.0)
n = u_test.shape[1]
u_train, u_test = u_train.reshape(-1, n, n), u_test.reshape(-1, n, n)
# %%
# plt.imshow(u_test[0, :].reshape(28, 28))

# %%

A = graph(1, 2, 1)

init_cond = np.zeros(A.shape[0])
init_cond[0] = 1
X, cache = reservoir(u_train, A, init_cond=init_cond)

#%%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 5000)
def predict(model, X, y):
    y_index = list(range(1, 10))
    model.predict(X, y)
model.fit(X, y_train)

#%%
# logisticRegression.predict(x_test[0].reshape(1,-1)
def accuracy(y, y_pred):
    return sum(y==y_pred)/y.shape[0]

y_pred = model.predict(X)
#%%
accuracy(y_pred, y_train)
#%%
W_r, W_in, init_cond = cache
X_test = prediction(u_test, W_r, W_in, init_cond)
y_test_pred = model.predict(X_test)

#%%
accuracy(y_test, y_test_pred)

#%%
# -------------------------------------------------------
# Using the iteneraty instead of the entire state vector
init_cond = np.zeros(A.shape[0])
init_cond[0] = 1
X, cache = reservoir_itin(u_train, A, init_cond=init_cond)

#%%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 2000)
model.fit(X, y_train)

#%%
# logisticRegression.predict(x_test[0].reshape(1,-1)
def accuracy(y, y_pred):
    return sum(y==y_pred)/y.shape[0]

y_pred = model.predict(X)
#%%
accuracy(y_pred, y_train)
#%%
W_r, W_in, init_cond = cache
X_test = prediction_itin(u_test, W_r, W_in, init_cond)
y_test_pred = model.predict(X_test)

#%%
accuracy(y_test, y_test_pred)

#%%
#------------------------------------------------------------
#  without reservoir
N, m, n = u_train.shape
lr_u_train = u_train.reshape(-1, m*n)
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(lr_u_train, y_train)
lr_y_pred = lr_model.predict(lr_u_train)
#%%
accuracy(lr_y_pred, y_train)
#%%
W_r, W_in, init_cond = cache
lr_u_test = u_test.reshape(-1, m*n)
lr_y_test_pred = lr_model.predict(lr_u_test)

#%%
accuracy(y_test, lr_y_test_pred)
# %%
