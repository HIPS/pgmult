"""
Simple Geweke test for the stick breaking multinomial correlated LDA model
"""

import numpy as np
import matplotlib.pyplot as plt
from pybasicbayes.util.text import progprint_xrange

from pgmult.lda import StickbreakingCorrelatedLDA
from scipy.sparse import csr_matrix

# def test_geweke_lda():
if __name__ == "__main__":
    N_iter = 5000
    T = 3           # Number of topics
    D = 10         # Number of documents
    V = 20          # Number of words
    N = 20         # Number of words per document
    alpha_beta = 1.0

    # Generate synthetic data
    data = np.random.poisson(2, (D,V))
    data = csr_matrix(data)

    # Sample a GP
    model = StickbreakingCorrelatedLDA(data, T, alpha_beta=alpha_beta)

    # Run a Geweke test
    thetas = []
    betas = []
    for itr in progprint_xrange(N_iter):
        # Resample the data
        model.generate(N, keep=True)

        # Resample the parameters
        model.resample()

        # Update our samples
        thetas.append(model.theta.copy())
        betas.append(model.beta.copy())

    # Check that the PG-Multinomial samples are distributed like the prior
    thetas = np.array(thetas)
    theta_mean = thetas.mean(0)
    theta_std  = thetas.std(0)

    betas = np.array(betas)
    beta_mean = betas.mean(0)
    beta_std  = betas.std(0)

    # Now sample from the prior for comparison
    print("Sampling from prior")
    from pybasicbayes.distributions import GaussianFixedMean
    from pgmult.utils import compute_uniform_mean_psi, psi_to_pi
    mu, sigma0 = compute_uniform_mean_psi(T)
    psis_prior = np.array(
        [GaussianFixedMean(mu=mu, lmbda_0=T * sigma0, nu_0=T).rvs(1)
         for _ in range(N_iter)])
    thetas_prior = psi_to_pi(psis_prior[:,0,:])
    betas_prior = np.random.dirichlet(alpha_beta*np.ones(V), size=(N_iter,))

    # print "Mean psi: ", psi_mean, " +- ", psi_std

    import pybasicbayes.util.general as general
    percentilecutoff = 5
    def plot_1d_scaled_quantiles(p1,p2,plot_midline=True):
        # scaled quantiles so that multiple calls line up
        p1.sort(), p2.sort() # NOTE: destructive! but that's cool
        xmin,xmax = general.scoreatpercentile(p1,percentilecutoff), \
                    general.scoreatpercentile(p1,100-percentilecutoff)
        ymin,ymax = general.scoreatpercentile(p2,percentilecutoff), \
                    general.scoreatpercentile(p2,100-percentilecutoff)
        plt.plot((p1-xmin)/(xmax-xmin),(p2-ymin)/(ymax-ymin))

        if plot_midline:
            plt.plot((0,1),(0,1),'k--')
        plt.axis((0,1,0,1))

    # Plot sample averages and compare to the prior
    plt.subplot(211)
    plot_1d_scaled_quantiles(betas[:,0,0], betas_prior[:,0])

    plt.subplot(212)
    plot_1d_scaled_quantiles(thetas[:,0,-1], thetas_prior[:,-1])
    plt.show()

    # Test dirichlet sampling
    from scipy.stats import beta, probplot
    marg_beta = beta(alpha_beta, alpha_beta*(V-1))
    probplot(betas_prior[:,0], dist=marg_beta, fit=False, plot=plt.subplot(111))
    plt.show()

    # # Plot sample averages and compare to the prior
    # plt.subplot(211)
    # plt.bar(np.arange(V), beta_mean[:,0], yerr=beta_std[:,0], color='r')
    # plt.plot(np.arange(V), 1./V*np.ones(V), ':k')
    # plt.xlabel("V")
    # plt.ylabel("$p(\\beta_0[v])$")
    #
    # plt.subplot(212)
    # plt.bar(np.arange(T), theta_mean[0], yerr=theta_std[0], color='r')
    # plt.plot(np.arange(T), 1./T*np.ones(T), ':k')
    # plt.xlabel("T")
    # plt.ylabel("$p(\\theta_0[t])$")


    plt.show()
