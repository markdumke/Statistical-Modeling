# -*- coding: utf-8 -*-
"""
K-means clustering

"""
import numpy as np

# Generate some data points
np.random.seed(210816)
mu1 = [5, 5]
mu2 = [2, 4]
sigma1 = np.diag([0.4, 0.2])
sigma2 = np.diag([0.4, 0.2])
x1 = np.random.multivariate_normal(mu1, sigma1, 60)
x2 = np.random.multivariate_normal(mu2, sigma2, 30)

x = np.concatenate((x1, x2), axis = 0)
x_label = np.zeros((len(x), 1))
x = np.column_stack([x, x_label]) # 3.column for class assignment

from matplotlib import pyplot as plt
plt.plot(x[:,0], x[:,1], "ro")
plt.show()

#---------------------
# k means clustering
k = 3

# start points
cl_0 = [1, 4]
cl_1 = [2, 3.5]

# iterate:  calculate distance from points to cluster centers cl_i
#           assign points to nearest cluster
#           calculate new mean for each cluster

# Euclidean distance
def dist(x, y):
    return(np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))

dist_cl = np.zeros([len(x), 2])
iter = 100
for i in range(iter):
    for j in range(len(x)):
        dist_cl[j, 0] = dist(x[j, 0:2], cl_0)
        dist_cl[j, 1] = dist(x[j, 0:2], cl_1)
        x[j, 2] = dist_cl[j, :].argmin()
    points_cl0 = x[np.where(x[:, 2] == 0), 0:2]
    points_cl1 = x[np.where(x[:, 2] == 1), 0:2]
    cl_0 = points_cl0.mean(axis = 1).T
    cl_1 = points_cl1.mean(axis = 1).T
    

# Plotting clusters
fig = plt.figure()
scatter = plt.scatter(x[:, 0], x[:, 1], c = x[:, 2], s = 100, alpha=0.5, cmap ='gray')
plt.plot(cl_0[0], cl_0[1], "rx", markersize = 20, markeredgewidth = 2)
plt.plot(cl_1[0], cl_1[1], "rx", markersize = 20, markeredgewidth = 2)
fig.show()