# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt

# Classification with AdaBoost

# sample some data from two normal distributions
np.random.seed(28082016)
Sigma_0 = np.array([[1, 0.5], [0.5, 1]])
x0 = np.random.multivariate_normal([1, 1], Sigma_0, 70)
Sigma_1 = np.array([[1, 0.5], [0.5, 1]])
x1 = np.random.multivariate_normal([2, 0], Sigma_1, 30)
x12 = np.random.multivariate_normal([-1, 2], Sigma_1, 40)

x_class = np.repeat([-1, 1], repeats = (len(x0), len(x1) + len(x12)))
X = np.vstack((x0, x1, x12))
X = np.hstack((X, x_class[:, np.newaxis]))

############################################################
# Classify points with AdaBoost
#   using decision trees (one split -> stumps) as base learners

def ada_boost(x, features, target, mstop):
    w = np.repeat(1 / len(target), len(target))   #initialize weights
    alpha = np.zeros(mstop)
    err = np.zeros(mstop)
    stump = DecisionTreeClassifier(max_depth = 1) # decision tree
    clf = np.repeat(stump, mstop)
    prediction = np.zeros((mstop, len(x)))

    for m in range(mstop):
# Fit base learner g_m (here: tree) to weighted training data w_i * x_i   
        clf[m] = clf[m].fit(features, target, sample_weight = w)

# Calculate weighted misclassification rate err_m
        misclassifed = target != clf[m].predict(features)
        err[m] = sum(w * misclassifed)
# Alternative: err = 1 - clf.score(features, target, sample_weight = w)

# Compute weight alpha_m of base learner
        alpha[m] = 1 / 2 * np.log((1 - err[m]) / err[m])

# Compute new weights w_i and normalize sum to 1
        w = w * np.exp(alpha[m] * misclassifed)
        w = w / sum(w)

# Predict x with model m
        prediction[m] = clf[m].predict(x).astype(np.float)

# Output: sign(sum(alpha_m * clf_m))
    return(np.sign(np.sum(prediction.T * alpha, axis = 1)))

# Test
ada_boost(X[:, 0:2], features = X[:, 0:2], target = X[:, 2], mstop = 30)

## Compare with build-in sklearn.ensemble.AdaBoostClassifier
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), 
#                         algorithm = "SAMME", n_estimators = 30)
#clf.fit(X[:, 0:2], y = X[:, 2])
#clf.predict(X[:, 0:2])

###################################
# Visualisation: Plot decision boundaries
plot_step = 0.02
class_names = "AB"

# Plot the decision boundaries
x0_min, x0_max = np.min(x0) - 1, np.max(x0) + 1
x1_min, x1_max = np.min(x1) - 1, np.max(x1) + 1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, plot_step),
                     np.arange(x1_min, x1_max, plot_step))

Z = ada_boost(np.c_[xx0.ravel(), xx1.ravel()], features = X[:, 0:2],
              target = X[:, 2], mstop = 500)
Z = Z.reshape(xx0.shape)
cs = plt.contourf(xx0, xx1, Z, cmap = plt.cm.Paired)
plt.axis("tight")

color = np.repeat(["blue", "red"], repeats = (len(x0), len(x1) + len(x12))) # for plots
X = np.hstack((X, color[:, np.newaxis]))

plt.scatter(X[:, 0], X[:, 1], c = X[:, 3])
plt.title("Two-class Classification")