# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import bernoulli

""" Naive Bayes Classifier 

# y binary labels -> Y ~ Ber(p)
# x binary features -> X_j|Y=c ~ Ber(theta_jc)

"""

# Training
# Maximum Likelihood Estimates for parameters of Bernoulli distributions
def naive_bayes_train(x, y):
    n_classes = len(np.unique(y))
    n_features = x.shape[1]
    p = np.zeros(n_classes)
    theta = np.zeros((n_classes, n_features, 2))
    for i in range(n_classes):
        p[i] = np.mean(y == i)
        for j in range(n_features):
            theta[i, j, 1] = np.mean(x[y == i, j])
    theta[:, :, 0] = 1 - theta[:, :, 1]        
    return(theta, p)

# Prediction
def naive_bayes_predict(x, theta, p, prob = False):
    # x_test vector
    i = np.arange(x.shape[1])
    prob_classes = np.zeros((len(x), len(p)))
    for j in range(len(x)):
        numerator = p * np.prod(theta[:, i, x[j]], axis = 1)
        prob_classes[j] = numerator / np.sum(numerator)
    if prob == True:
        return(prob_classes)
    else:
        return(np.argmax(prob_classes, axis = 1))
# 0 probablities can occur, if a count is zero
        
# Example
N = 100
n_features = 5
y = bernoulli.rvs(0.5, size = N)
x_train = bernoulli.rvs(0.6, size = (N, n_features))
x_test = bernoulli.rvs(0.5, size = (20, n_features))

theta, p = naive_bayes_train(x_train, y)
naive_bayes_predict(x_test, theta, p)
proba = naive_bayes_predict(x_test, theta, p, prob = True)

# Comparison with sklearn implementation of Naive Bayes
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha = 0)
clf.fit(x_train, y)
clf.predict(x_test)

np.max(clf.predict_proba(x_test) - proba)