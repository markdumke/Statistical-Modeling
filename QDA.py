# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, (0, 3)]
y = iris.target

# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Visualisation: Plot decision boundaries and points
x0_min, x0_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
x1_min, x1_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, 0.02),
                     np.arange(x1_min, x1_max, 0.02))

Z = lda.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
plt.contourf(xx0, xx1, Z, cmap = plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.title("Linear Discriminant Analysis")
plt.show()

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)

Z = qda.predict(np.c_[xx0.ravel(), xx1.ravel()])
Z = Z.reshape(xx0.shape)
plt.contourf(xx0, xx1, Z, cmap = plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c = y)
plt.title("Quadratic Discriminant Analysis")
plt.show()