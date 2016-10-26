"""
Simple Geweke test for the multinomial GP
"""

import numpy as np
import matplotlib.pyplot as plt

from GPy.kern import RBF
from pgmult.gp import MultinomialGP
from pgmult.utils import psi_to_pi, N_vec, kappa_vec

def geweke_test(K, N_iter=10000):
    D = 1           # Input dimensionality
    M = 20         # Number of observed datapoints
    l = 10.0        # Length scale of GP
    L = 100.0       # Length of observation sequence
    v = 1.0         # Variance of the GP

    # Initialize a grid of points at which to observe GP
    N_max = 10
    Z = np.linspace(0,L,M)[:,None]

    # Initialize the kernel
    kernel = RBF(1, lengthscale=l, variance=v)

    # Sample a GP
    model = MultinomialGP(K, kernel, D=D)
    X, psi = model.generate(Z=Z, N=N_max*np.ones(M, dtype=np.int), keep=True, full_output=True)
    data = model.data_list[0]
    data["psi"] = psi
    model.resample_omega()

    # Helper function to resample counts
    def _resample_X():
        pis = model.pi(data)
        X = np.array([np.random.multinomial(N_max, pis[m]) for m in range(M)])
        N = N_vec(X).astype(np.float)
        kappa = kappa_vec(X)

        data["X"] = X
        data["N"] = N
        data["kappa"] = kappa

    # Run a Geweke test
    Xs = []
    psi_samples = []

    # samples = []
    for itr in range(N_iter):
        if itr % 10 == 0:
            print("Iteration ", itr)

        # Resample the data
        _resample_X()

        # Resample the PG-Multinomial parameters
        model.resample_model()

        # Update our samples
        Xs.append(data["X"].copy())
        psi_samples.append(data["psi"].copy())

    # Check that the PG-Multinomial samples are distributed like the prior
    psi_samples = np.array(psi_samples)
    psi_mean = psi_samples.mean(0)
    psi_std  = psi_samples.std(0)
    print("Mean psi: ", psi_mean, " +- ", psi_std)

    # Plot sampled psi vs prior mean and variances
    fig, axs = plt.subplots(K-1, 1)
    for k,ax in enumerate(axs):
        # Plot the empirical sample distribution
        ax.errorbar(Z, psi_mean[:,k], yerr=psi_std[:,k], color='b', lw=2)

        # Plot the prior
        ax.plot(Z, np.zeros_like(Z), '-k')
        ax.plot(Z, np.ones_like(Z), ':k')
        ax.plot(Z, -np.ones_like(Z), ':k')

        ax.set_xlabel("Z")
        ax.set_ylabel("$\\psi_{%d}$" % k)

    plt.ioff()
    plt.show()

geweke_test(5)
