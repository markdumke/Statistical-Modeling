# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import beta
from scipy.stats import uniform
import matplotlib.pyplot as plt
# Rejection Sampling

"""
Goal: generate random numbers from f(x)
Iterate:
    Generate random number x from envelope f(y)
    Calculate acceptance probability alpha = gamma * f(x) / f(y)
    Generate uniform random number in [0, 1]
    If x < alpha -> accept random number as random number from f(x)
"""

# Simple example
# Generate n random numbers from Beta(a, b)
# Envelope: f(y) uniform distribution

# Parameters of beta distribution
a = 2
b = 2

def fx(x, a, b):
    return(beta.pdf(x, a = a, b = b))

def fy(x):
    return(uniform.pdf(x))

gamma = 1 / 1.5 # factor for the envelope because max(beta.pdf(x, 2, 2)) = 1.5 for x = 0.5

n = 2000 # number of random numbers
x = np.zeros(n)
i = 0
while i < n:
    y = np.random.random(1)
    alpha = gamma * fx(y, a = a, b = b) / fy(y)
    u = np.random.random(1)
    if u < alpha:
        x[i] = y
        i = i + 1

x_space = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
plt.plot(x_space, beta.pdf(x_space, a, b), 'r-', lw = 3, alpha = 0.6, 
         label = 'beta pdf')
plt.plot(x_space, 1 / gamma * uniform.pdf(x_space), 'b-', lw = 3, alpha = 0.6, 
         label = 'uniform envelope')
plt.title("Rejection Sampling for Beta(2, 2)")
plt.legend(loc = 10)
plt.show()
plt.hist(x, bins = 20)