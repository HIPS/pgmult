"""
Test the LDS held out likelihoods against a Monte Carlo comparison
"""

import numpy as np
from scipy.misc import logsumexp

from pgmult.lds import MultinomialLDS
from pybasicbayes.distributions import Gaussian
from autoregressive.distributions import AutoRegression

np.seterr(invalid="warn")
np.random.seed(0)

#  set true params of the LDS
N = 1       # Number of events per time bin
T = 100       # Number of time bins
D = 2       # Latent state dimensionality
mu_init = np.random.randn(D)
sigma_init = 0.1*np.eye(D)

A = 0.99 * np.eye(D)
A[:2,:2] = \
    0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.1*np.eye(D)

K = 3
# C = np.hstack((np.ones((K-1, 1)), np.zeros((K-1, D-1))))
C = np.random.randn(K-1, D)

###################
#  generate data  #
###################

model = MultinomialLDS(K, D,
    init_dynamics_distn=Gaussian(mu=mu_init,sigma=sigma_init),
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    C=C
    )
data = model.generate(T=T, N=N, keep=False)
# data["x"] = np.hstack([np.zeros((T,K-1)), np.ones((T,1))])

# Estimate the held out likelihood using Monte Carlo
M = 10000
hll_mc, std_mc = model._mc_heldout_log_likelihood(data["x"], M=M)

# Estimate the held out log likelihood
# hll_info, std_info = model._info_form_heldout_log_likelihood(data["x"], M=M)
hll_dist, std_dist = model._distn_form_heldout_log_likelihood(data["x"], M=M)

print("MC. Model: ", hll_mc, " +- ", std_mc)
# print "AN. Model (info): ", hll_info, " +- ", std_info
print("AN. Model (dist): ", hll_dist, " +- ", std_dist)
