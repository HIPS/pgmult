"""
Simple Geweke test for the PG-augmented Multinomial distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot

from pgmult.distributions import PGLogisticNormalMultinomial

def geweke_test(K, N_iter=10000):
    """
    """
    # Create a multinomial distribution
    mu = np.zeros(K)
    mu[-1] = 1
    Sigma = np.eye(K)
    pgm = PGLogisticNormalMultinomial(K, mu=mu, Sigma=Sigma)

    # Run a Geweke test
    xs = []
    samples = []
    for itr in range(N_iter):
        if itr % 10 == 0:
            print("Iteration ", itr)
        # Resample the data
        x = pgm.rvs(1)

        # Resample the PG-Multinomial parameters
        pgm.resample(x)

        # Update our samples
        xs.append(x.copy())
        samples.append(pgm.copy_sample())

    # Check that the PG-Multinomial samples are distributed like the prior
    psi_samples = np.array([s.psi for s in samples])
    psi_mean = psi_samples.mean(0)
    psi_std  = psi_samples.std(0)
    print("Mean bias: ", psi_mean, " +- ", psi_std)

    # Make Q-Q plots
    ind = K-2
    fig = plt.figure()
    ax = fig.add_subplot(121)
    psi_dist = norm(mu[ind], np.sqrt(Sigma[ind,ind]))
    probplot(psi_samples[:,ind], dist=psi_dist, plot=ax)

    fig.add_subplot(122)
    _, bins, _ = plt.hist(psi_samples[:,ind], 20, normed=True, alpha=0.2)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    plt.plot(bincenters, psi_dist.pdf(bincenters), 'r--', linewidth=1)
    plt.show()

geweke_test(4)