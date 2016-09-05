# -*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt

boston = datasets.load_boston()
X = boston.data
y = boston.target

# Ordinary Least Squares
lm = linear_model.LinearRegression()
lm.fit(X, y)
lm.score(X, y) # R^2 = 0.741
lm.coef_
lm.intercept_

## Ridge Regression (L2 penalty)
alphas = np.logspace(-6, 6, 100)

ridge = linear_model.RidgeCV(alphas = alphas)
ridge.fit(X, y)
ridge.alpha_
ridge.coef_

# Lasso (L1 penalty)
lasso = linear_model.LassoCV(alphas = alphas)
lasso.fit(X, y)
lasso.alpha_
lasso.coef_

# Elastic Net (convex combination of Lasso and Ridge)
enet = linear_model.ElasticNetCV(alphas = alphas, 
                                 l1_ratio = [.1, .5, .7, .9, .95, .99, 1])
enet.fit(X, y)
enet.alpha_
enet.l1_ratio_
enet.coef_

# Comparison LM - Ridge - Lasso - Elastic Net
ncoef = len(lasso.coef_)
x = np.arange(0, 5 * len(lasso.coef_), step = 5)
plt.plot(x, lm.coef_, "ko", label = "LM")
plt.plot(x + 1, ridge.coef_, "bo", label = "Ridge")
plt.plot(x + 2, lasso.coef_, "yo", label = "Lasso")
plt.plot(x + 3, enet.coef_, "ro", label = "Elastic Net")
plt.title("Coefficients of linear models")
plt.legend(loc = 10)
plt.show()

# Robust regression (outliers)

# generate some data
np.random.seed(5092016)
x = np.random.random((30, 1))
y = 2 * x + np.random.normal(size = (30, 1))
it = np.random.choice(np.arange(len(y)), size = 4, replace = False)
y[it] = y[it] + 15 # outliers

# Linear Regression fit
lm = linear_model.LinearRegression()
lm.fit(x, y)
lm.coef_

# Robust regression 
robust = linear_model.RANSACRegressor()
robust.fit(x, y)
robust.estimator_.coef_

plt.plot(x, y, "ko", markersize = 4)
plt.ylim(- 2, 20)
plt.xlim(- 0.1, 1.1)
plt.plot(x, lm.intercept_ + lm.coef_ * x, linewidth = 2, label = "LM")
plt.plot(x, robust.estimator_.intercept_ + robust.estimator_.coef_ * x, 
         linewidth = 2, label = "Robust")
plt.legend(loc = 2)
plt.show()
