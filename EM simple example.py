# -*- coding: utf-8 -*-
import numpy as np
'''
EM Algorithm for simple example

data in table:    x11 x12 x13 ...
                  x21 x22 NA ...  with some missing values (NA)

Model: xij = mu + alpha_i + beta_j + error_ij
    with alpha_i effect of row i and beta_j effect of column j, mu overall mean
    and Sum(alpha_i) = 0, Sum(beta_j) = 0
    
EM Algorithm:   iterate between Expectation (e.g. calculate missing values) 
                and Maximisation (find good parameters for model)
'''

data = np.array([[10, 15, 17], [22, 23, None]])
missings = np.equal(data, None)

# initialisation
data[missings] = 0 # all missing values replaced by 0

iter = 20
nrow = np.shape(data)[0]
ncol = np.shape(data)[1]
alpha = np.zeros((iter, nrow))
beta = np.zeros((iter, ncol))
mu = np.zeros(iter)
imputations = np.zeros(iter)

# iterate E- and M-Step
for s in range(iter):
    # M-Step:
    mu[s] = np.mean(data)
    for i in range(nrow):
        alpha[s, i] = 1 / ncol * sum(data[i, :]) - mu[s]
    for j in range(ncol):
        beta[s, j] = 1 / nrow * sum(data[:, j]) - mu[s]
    
    # E-Step
    for i in range(nrow):
        for j in range(ncol):
            if missings[i, j] == True:
                data[i, j] = mu[s] + alpha[s, i] + beta[s, j]
                imputations[s] = data[i, j]
            
# some fancy plots
from matplotlib import pyplot as plt
plt.plot(mu, "ro-", label = r'$\mu$')
plt.plot(alpha[:, 0], "bo-", label = r'$\alpha_0$')
plt.title("EM M-Step")
plt.ylabel("Estimate for parameters")
plt.xlabel("Iterations")
plt.legend()
plt.show()

plt.plot(imputations, "go-")
plt.title("EM E-Step")
plt.ylabel("Imputed value")
plt.xlabel("Iterations")