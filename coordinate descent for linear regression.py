# -*- coding: utf-8 -*-
"""
Coordinate Descent for linear regression

"""
import numpy as np
import scipy as sp
from scipy import optimize

# generate some data
np.random.seed(200816)
x = np.random.random(20)
def true_function(x):
    return(2 * x)
y = true_function(x) + np.random.normal(0, 0.5, len(x))

from matplotlib import pyplot as plt
plt.plot(x, y, "ro")

# using coordinate descent to find estimates for linear regression coefficients
n = len(x)
beta = [0, 0] # initialisation

iter = 100 # number of iterations
beta = np.zeros((iter, 2))

def loss(beta_0, beta_1):
    return(((beta_0 + x * beta_1 - y)**2).mean())

for i in range(1, 100):
    beta[i, 0] = sp.optimize.minimize_scalar(loss(beta[i, 0], beta[i - 1, 1]))
    beta[i, 1] = sp.optimize.minimize_scalar(loss(beta[i, 0], beta[i, 1]))

# calculate solution for beta directly using normal equation
# design matrix X with one-column for bias beta[0]
X = np.vstack([np.ones_like(x), x]).T
beta_direct = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T) , y)

print(beta[- 1, ])
print(beta_direct)

# plot data generating process and estimated regression line
x_true = np.linspace(0, 1, 50)
y_true = 2 * x_true

plt.plot(x_true, y_true, "-r", label = "true mean")
plt.title("Linear Regression")

y_hat = beta[-1, 0] + x_true * beta[-1, 1]
plt.plot(x_true, y_hat, "-b", label = "linear regression")
plt.legend()
plt.show()

# plotting loss of the gradient descent
y_pred = np.zeros((iter, n))
L = np.zeros(iter)
for i in range(iter):
    y_pred[i, :] = beta[i, 0] + x * beta[i, 1]
    L[i] = 1/n * sum((y_pred[i, :] - y)**2)

plt.plot(beta[:, 1], L, "-ro")
plt.ylabel("Loss")
plt.xlabel("beta_1")
plt.show()

#def loss(beta_0, beta_1):
#    return(((beta_0 + x * beta_1 - y)**2).mean())

#delta = 0.25
#a = np.arange(-3.0, 3.0, delta)
#b = np.arange(-2.0, 2.0, delta)
#X, Y = np.meshgrid(a, b)
#Z = loss(X, Y)
#plt.contour(X,Y,Z)
