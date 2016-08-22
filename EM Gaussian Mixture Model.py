# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import multivariate_normal
'''
EM Algorithm for Gaussian Mixture Model

data (x, z), where xi are observed data points and z is a latent variable 
  indicating from which unviariate Gaussian (cluster) xi is
  
Model: f(x, z) =  sum(pi_k**z * f_k(x)) with f_k(x) = N(mu_k, Sigma_k) mult. Normal
'''
# Sample some data_points x = (x1, x2) from multivariate normal
np.random.seed(220816)
n0 = 20
mu_0_true = np.array([0, 4])
Sigma_0_true = np.array([[0.4, 0.7], [0.7, 5]])
cl_0 = np.random.multivariate_normal(mu_0_true, Sigma_0_true, n0)

n1 = 15
mu_1_true = np.array([- 5, 0])
Sigma_1_true = np.array([[2, 0.3], [0.3, 0.7]])
cl_1 = np.random.multivariate_normal(mu_1_true, Sigma_1_true, n1)

x = np.vstack((cl_0, cl_1))

from matplotlib import pyplot as plt
#plt.plot(x[:, 0], x[:, 1], "ro")
plt.plot(x[0:n0, 0], x[0:n0, 1], "bo")
plt.plot(x[n0:(n0 + n1), 0], x[n0:(n0 + n1), 1], "go")

# Initialise parameter for each cluster k, we choose K = 2 here
K = 2
mu = np.array([[np.random.normal(size = 2)],[np.random.normal(size = 2)]])
Sigma = np.array([[np.eye(2)],[np.eye(2)]])
pi = np.zeros(2)
pi[0] = np.random.random(size = 1)
pi[1] = 1 - pi[0]

def g(x, pi, mu_0, mu_1, Sigma_0, Sigma_1): 
    return(pi[0] * multivariate_normal.pdf(x, mean = mu_0, cov = Sigma_0) +
           pi[1] * multivariate_normal.pdf(x, mean = mu_1, cov = Sigma_1))

r = np.zeros((len(x), K))
m_ = np.zeros(K)
S = np.zeros((np.shape(x)[0], 2, 2))

# EM: Iterate E-step and M-step
iter = 10
def em(iter):
    for i in range(iter):
        # E step
        # compute soft clustering of points xi to f_1 and f_2
        r[:, 0] = (pi[0] * multivariate_normal.pdf(x, mean = mu[0, :][0], cov = Sigma[0, :][0]) /
                    g(x, pi, mu[0, :][0], mu[1, :][0], Sigma[0, :][0] , Sigma[1, :][0]))
        r[:, 1] = 1 - r[:, 0]
    
        # M step
        # estimate the MLE for parameters mu, Sigma, pi
        m = np.sum(r, axis = None)
        for k in range(K):
            m_[k] = np.sum(r[:, k])
            pi[k] = m_[k] / m
            mu[k] = 1 / m_[k] * np.dot(r[:, k].T, x)      
            for j in range(np.shape(x)[0]):
                S[j] = np.dot((x[j, :] - mu[k]).T, (x[j, :] - mu[k])) * r[j, k]
            Sigma[k] = 1 / m_[k] * sum(S)    
    return(pi, mu, Sigma)
    
em_1 = em(1)
mu_1 = em_1[1]
plt.plot(mu_1[0, 0][0], mu_1[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_1[1, 0][0], mu_1[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)

em_2 = em(2)
mu_2 = em_2[1]
plt.plot(mu_2[0, 0][0], mu_2[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_2[1, 0][0], mu_2[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)

em_3 = em(3)
mu_3 = em_3[1]
plt.plot(mu_3[0, 0][0], mu_3[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_3[1, 0][0], mu_3[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)

em_5 = em(5)
mu_5 = em_5[1]
plt.plot(mu_5[0, 0][0], mu_5[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_5[1, 0][0], mu_5[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)

em_10 = em(10)
mu_10 = em_10[1]
plt.plot(mu_10[0, 0][0], mu_10[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_10[1, 0][0], mu_10[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)

em_100 = em(100)
mu_100 = em_100[1]
plt.plot(mu_100[0, 0][0], mu_100[0, 0][1], "bx", markersize = 20, markeredgewidth = 2)
plt.plot(mu_100[1, 0][0], mu_100[1, 0][1], "gx", markersize = 20, markeredgewidth = 2)
plt.title("EM Algorithm for Gaussian Mixture Model")
plt.xlabel("x")
plt.ylabel("y")
