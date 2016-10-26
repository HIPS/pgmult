"""
Linear dynamical system model for the AP text dataset.
Each document is modeled as a draw from an LDS with
categorical observations.
"""

import os
import gzip
import time
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from hips.plotting.layout import create_axis_at_location, create_figure
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors

from pgmult.lds import MultinomialLDS
from pgmult.hmm import MultinomialHMM

from pybasicbayes.distributions import Gaussian, AutoRegression
from pybasicbayes.util.text import progprint_xrange

from pylds.models import DefaultLDS

np.seterr(invalid="warn")
np.random.seed(0)

# Model parameters
D = 10      # Latent dimension
K = 100    # Number of words
T = 20    # Number of time bins
N = 1    # Number of events per multinomial sample
Ndata = 2  # Number of datasets

def make_synthetic_data():
    mu_init = np.zeros(D)
    # mu_init[0] = 1.0
    sigma_init = 0.5*np.eye(D)

    A = np.eye(D)
    # A[:2,:2] = \
    #     0.99*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
    #                    [np.sin(np.pi/24),  np.cos(np.pi/24)]])
    sigma_states = 0.1*np.eye(D)

    # C = np.hstack((np.ones((K-1, 1)), np.zeros((K-1, D-1))))
    C = 0. * np.random.randn(K-1, D)

    truemodel = MultinomialLDS(K, D,
        init_dynamics_distn=Gaussian(mu=mu_init,sigma=sigma_init),
        dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
        C=C
        )

    data_list = []
    Xs = []
    for i in range(Ndata):
        data = truemodel.generate(T=T, N=N)
        data_list.append(data)
        Xs.append(data["x"])
    return data_list, Xs

# Inference
def fit_gaussian_lds_model(Xs, N_samples=100):
    testmodel = DefaultLDS(n=D,p=K)

    for X in Xs:
        testmodel.add_data(X)

    samples = []
    lls = []
    for smpl in progprint_xrange(N_samples):
        testmodel.resample_model()

        samples.append(testmodel.copy_sample())
        lls.append(testmodel.log_likelihood())

    lls = np.array(lls)
    return lls

def fit_lds_model(Xs, Xtest, N_samples=100):
    model = MultinomialLDS(K, D,
        init_dynamics_distn=Gaussian(mu_0=np.zeros(D), sigma_0=np.eye(D), kappa_0=1.0, nu_0=D+1.0),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=np.eye(D),M_0=np.zeros((D,D)),K_0=np.eye(D)),
        sigma_C=1
        )

    for X in Xs:
        model.add_data(X)
    data = model.data_list[0]

    samples = []
    lls = []
    test_lls = []
    mc_test_lls = []
    pis = []
    psis = []
    zs = []
    timestamps = [time.time()]
    for smpl in progprint_xrange(N_samples):
        model.resample_model()
        timestamps.append(time.time())

        samples.append(model.copy_sample())
        # TODO: Use log_likelihood() to marginalize over z
        lls.append(model.log_likelihood())
        # test_lls.append(model.heldout_log_likelihood(Xtest, M=50)[0])
        mc_test_lls.append(model._mc_heldout_log_likelihood(Xtest, M=1)[0])
        pis.append(model.pi(data))
        psis.append(model.psi(data))
        zs.append(data["states"].stateseq)

    lls = np.array(lls)
    test_lls = np.array(test_lls)
    pis = np.array(pis)
    psis = np.array(psis)
    zs = np.array(zs)
    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]
    return model, lls, test_lls, mc_test_lls, pis, psis, zs, timestamps

def fit_hmm(Xs, Xtest, N_samples=100):
    model = MultinomialHMM(K, D)

    for X in Xs:
        model.add_data(X)

    samples = []
    lls = []
    test_lls = []
    pis = []
    zs = []
    timestamps = [time.time()]
    for smpl in progprint_xrange(N_samples):
        model.resample_model()
        timestamps.append(time.time())

        samples.append(model.copy_sample())
        # TODO: Use log_likelihood() to marginalize over z
        lls.append(model.log_likelihood())
        # lls.append(model.log_likelihood_fixed_z())
        test_lls.append(model.log_likelihood(Xtest))
        # pis.append(testmodel.pis()[0])
        zs.append(model.stateseqs[0])

    lls = np.array(lls)
    test_lls = np.array(test_lls)
    pis = np.array(pis)
    zs = np.array(zs)
    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]

    return model, lls, test_lls, pis, zs, timestamps

def plot_log_likelihood(lds_times, lds_lls,
                        hmm_times, hmm_lls,
                        figname="lds_vs_hmm_ll.pdf"):
    # Plot the log likelihood
    plt.figure(figsize=(3,3.2))
    plt.plot(hmm_lls, lw=2, color=colors[1], label="HMM")
    plt.plot(lds_lls, lw=2, color=colors[0], label="LDS")

    plt.legend(loc="lower right")
    plt.xlabel('Iteration')
    plt.ylabel("Log Likelihood")
    plt.title("Synthetic. D=%d. K=%d" % (D,K))
    # plt.savefig(os.path.join("results", "ap", figname))

    plt.tight_layout()


def plot_lds_results(X, z_mean, z_std, pis):
    # Plot the true and inferred states
    plt.figure()
    ax1 = plt.subplot(311)
    plt.errorbar(z_mean[:,0], color="r", yerr=z_std[:,0])
    plt.errorbar(z_mean[:,1], ls="--", color="r", yerr=z_std[:,1])
    ax1.set_title("True and inferred latent states")

    ax2 = plt.subplot(312)
    plt.imshow(X.T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    ax2.set_title("Observed counts")

    ax4 = plt.subplot(313)
    N_samples = pis.shape[0]
    plt.imshow(pis[N_samples//2:,...].mean(0).T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
    ax4.set_title("Mean inferred probabilities")
    plt.show()

if __name__ == "__main__":
    datas, Xs = make_synthetic_data()
    Xtrain = Xs[:-1]
    Xtest = Xs[-1]
    # K = len(key)

    N_samples = 25

    # Fit the Multinomial LDS
    print("Fitting multinomial LDS model")
    lds_model, lds_lls, lds_test_lls, lds_mc_test_lls, lds_pis, lds_psis, lds_zs, lds_times = \
        fit_lds_model(Xtrain, Xtest, N_samples=N_samples)
    # lds_psi_mean = lds_psis[N_samples//2:,...].mean(0)
    # lds_psi_std = lds_psis[N_samples//2:,...].std(0)
    # lds_z_mean = lds_zs[N_samples//2:,...].mean(0)
    # lds_z_std = lds_zs[N_samples//2:,...].std(0)

    # Fit the HMM
    # print "Fitting HMM"
    # hmm, hmm_lls, hmm_test_lls, hmm_pis, hmm_zs, hmm_times =\
    #     fit_hmm(Xtrain, Xtest, N_samples=N_samples)
    # hmm_pi_mean = hmm_pis[N_samples//2:,...].mean(0)
    # hmm_pi_std = hmm_pis[N_samples//2:,...].std(0)
    # hmm_z_mean = hmm_zs[N_samples//2:,...].mean(0)

    # Fit a Gaussian LDS
    # gauss_lds_lls = fit_gaussian_lds_model(Xs, N_samples=N_samples)

    # plot_qualitative_results(Xs[0], key, lds_psis[-1], lds_zs[-1])

    # Plot the predictive log likelihoods
    # plot_log_likelihood(lds_times, lds_lls,
    #                     hmm_times, hmm_lls,
    #                     figname="lds_vs_hmm_train_ll.pdf")

    # plot_log_likelihood(lds_times, lds_test_lls,
    #                     hmm_times, hmm_test_lls,
    #                     figname="lds_vs_hmm_test_ll.pdf")

    # plot_log_likelihood(lds_times, lds_test_lls,
    #                     lds_times, lds_mc_test_lls,
    #                     figname="lds_vs_hmm_test_ll.pdf")

    plt.figure(figsize=(3,3.2))
    plt.plot(lds_mc_test_lls, lw=2, color=colors[1], label="LDS MC")
    plt.plot(lds_test_lls, lw=2, color=colors[0], label="LDS Analytic")

    plt.legend(loc="lower right")
    plt.xlabel('Iteration')
    plt.ylabel("Log Likelihood")
    plt.title("Synthetic. D=%d. K=%d" % (D,K))


    plt.show()

