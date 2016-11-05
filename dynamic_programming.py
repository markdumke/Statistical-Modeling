# -*- coding: utf-8 -*-
import numpy as np 

# Implementation of Dynamic Programming to solve simple Markov Decision Process

# example Markov Process
# State Transition Probability Matrix P
P = np.array([[0.5, 0.2, 0.3, 0, 0], [0, 0, 0.5, 0.5, 0], [0, 0.1, 0.5, 0.4, 0], 
              [0.1, 0, 0.3, 0, 0.6], [0, 0, 0, 0, 1]])

# Sample from Markov Process:
# P with terminal state in last row
def sample_markov_process(prob):
    S = np.array([0])
    t = 0
    while S[t] != (len(P) - 1):
        S = np.append(S, np.random.choice(5, size = 1, p = prob[S[t - 1]]))
        t = t + 1
    return(S)

sample_markov_process(prob = P)

# Markov Reward Process
# Reward for each state
R = np.array([-1, -1, -1, -1, 0])

def sample_markov_reward_process(prob, r):
    S = sample_markov_process(prob)
    return(r[S])

sample_markov_reward_process(P, R)

# Markov Decision Process
# Example:  simple grid world, actions: going left, right, up, down, 
#           each transition gets reward of -1, final state, thereafter 0 rewards, 
#           if borders would be crossed, state is not changed
#           transitions with probability 1


# Policy Evaluation
# evaluate random policy pi, chooses each action with equal probabilty

# state value function v_pi
niter = 100
v = np.zeros((niter, 4, 1)) # initialisation
R = np.array([[-1], [-1], [-1], [0]])
gamma = 1 # discount factor
# transition probabilities for policy pi:
P = np.array([[0.5, 0.25, 0.25, 0], [0.25, 0.5, 0, 0.25], 
              [0.25, 0, 0.5, 0.25], [0, 0, 0, 1]]) 
for i in range(niter//2):
    v[i] = R + gamma * np.dot(P, v[i-1])
v # converges to true value function [-8, -6, -6, 0]

# greedy policy with respect to v[2] already optimal, 
#  so all subsequent steps unnecessary

# Policy Iteration = policy evaluation + greedy policy improvement
# Start with random policy pi_0
# evaluate v(pi_0) till convergence
# greedy policy improvement
P_1 = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 1], 
              [0, 0, 0, 1], [0, 0, 0, 1]])
# evaluate pi_1
for i in range(niter//2, niter):
    v[i] = R + gamma * np.dot(P_1, v[i-1])
v

# Value Iteration: just one step of policy evaluation, then greedy update of v
v = np.zeros((4, 1)) # initialisation
# transition probabilities for policy pi_0:
P = np.array([[0.5, 0.25, 0.25, 0], [0.25, 0.5, 0., 0.25], 
              [0.25, 0, 0.5, 0.25], [0, 0, 0, 1]])   
v = R + gamma * np.dot(P, v)
v
P = np.array([[0.5, 0.25, 0.25, 0], [0, 0, 0, 1], 
              [0, 0, 0, 1], [0, 0, 0, 1]])
v = R + gamma * np.dot(P, v)
v
P = np.array([[0, 0.5, 0.5, 0], [0, 0, 0, 1], 
              [0, 0, 0, 1], [0, 0, 0, 1]])
v = R + gamma * np.dot(P, v)
v







    