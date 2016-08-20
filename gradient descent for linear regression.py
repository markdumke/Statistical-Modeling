# -*- coding: utf-8 -*-
"""
Gradient Descent for linear regression

"""
import numpy as np

# generate some data
np.random.seed(200816)
x = np.random.random(20)
def true_function(x):
    return(2 * x)
y = true_function(x) + np.random.normal(0, 0.5, len(x))

from matplotlib import pyplot as plt
plt.plot(x, y, "ro")

# using gradient descent to find estimates for linear regression coefficients
n = len(x)
beta = [0, 0] # initialisation
epsilon = 1 # learning rate


def Grad_0(beta_0, beta_1):
    return(1 / n * (- sum(y) + n * beta[0] + beta[1] * sum(x)))

def Grad_1(beta_0, beta_1):
    return(1 / n * (- sum(y * x) + beta[0] * sum(x) + beta[1] * sum(x**2)))

beta = np.zeros((100, 2))
for i in range(1, 100):
    beta[i, 0] = beta[i - 1, 0] - epsilon * Grad_0(beta[i - 1, 0], beta[i - 1, 1])
    beta[i, 1] = beta[i - 1, 1] - epsilon * Grad_1(beta[i - 1, 0], beta[i - 1, 1])

# calculate solution for beta directly using normal equation
# design matrix X with one-column for bias beta[0]
X = np.vstack([np.ones_like(x), x]).T
beta_direct = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T) , y)

print(beta)
print(beta_direct)

# plot data generating process and estimated regression line
x_true = np.linspace(0, 1, 50)
y_true = 2 * x_true

plt.plot(x_true, y_true, "-r", label = "true mean")
plt.title("Linear Regression")

y_hat = beta[0] + x_true * beta[1]
plt.plot(x_true, y_hat, "-b", label = "linear regression")
plt.legend()
plt.show()




# Coordinate Descent for linear regression
