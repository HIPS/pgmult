"""
1D GP with multinomial observations
"""

import os
import time
from collections import namedtuple
import numpy as np
from GPy.kern import RBF
# np.random.seed(1122122122)

from pgmult.gp import MultinomialGP, LogisticNormalGP, EmpiricalStickBreakingGPModel
from pgmult.utils import psi_to_pi, compute_uniform_mean_psi

import matplotlib.pyplot as plt
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors


K = 4           # Size of output observations
N_max = 10      # Number of observations per input

def initialize_test(N_max=10, true_model_class=MultinomialGP):
    D = 1           # Input dimensionality

    M_train = 100   # Number of observed training datapoints
    M_test = 20     # Number of observed test datapoints
    M = M_train + M_test
    l = 10.0        # Length scale of GP
    L = 120.0       # Length of observation sequence
    v = 1.0         # Variance of the GP

    # Initialize a grid of points at which to observe GP
    N = N_max * np.ones(M, dtype=np.int32)
    Z = np.linspace(0,L,M)[:,None]

    # Initialize the kernel
    kernel = RBF(1, lengthscale=l, variance=v)

    # Sample a GP
    true_model = true_model_class(K, kernel, D=D)
    X, psi = true_model.generate(Z=Z, N=N, full_output=True)
    pi = np.array([psi_to_pi(p) for p in psi])

    # Split the data into training and test
    Dataset = namedtuple("Dataset", ["K", "kernel", "Z", "X", "psi", "pi"])
    train = Dataset(K, kernel, Z[:M_train], X[:M_train], psi[:M_train], pi[:M_train])
    test = Dataset(K, kernel, Z[M_train:], X[M_train:], psi[M_train:], pi[M_train:])

    return train, test

def initialize_interactive_plot(model, train, test):
    plot_K = isinstance(model, LogisticNormalGP)

    # Make predictions at the training and testing data
    pi_train, psi_train, _ = \
            model.collapsed_predict(train.Z)

    pi_test, psi_test, _ = \
            model.collapsed_predict(test.Z)

    lim = 5

    # PLOT!
    plt.ion()
    fig, axs = plt.subplots(train.K, 2)
    lns = np.zeros((train.K,4), dtype=object)
    for k in range(train.K):
        if k == train.K-1 and not plot_K:
            pass
        else:
            ax = axs[k,0]

            # Plot the training data
            ax.plot(train.Z, train.psi[:,k], '-b', lw=2)
            lns[k,0] = ax.plot(train.Z, psi_train[:,k], '--b', lw=2)[0]

            # Plot the testing data
            ax.plot(test.Z, test.psi[:,k], '-r', lw=2)
            lns[k,1] = ax.plot(test.Z, psi_test[:,k], '--r', lw=2)[0]

            # Plot the zero line
            ax.plot(train.Z, np.zeros_like(train.Z), ':k')
            ax.plot(test.Z, np.zeros_like(test.Z), ':k')
            # ax.set_xlim(0, L)
            ax.set_ylim(-lim, lim)
            ax.set_title("$\psi_%d$" % (k+1))

        ax = axs[k,1]
        pi_emp_train = train.X / train.X.sum(axis=1).astype(np.float)[:,None]
        ax.bar(train.Z, pi_emp_train[:,k], width=1, color='k')
        ax.plot(train.Z, train.pi[:,k], '-b', lw=2)
        lns[k,2] = ax.plot(train.Z, pi_train[:,k], '--b', lw=2)[0]

        pi_emp_test = test.X / test.X.sum(axis=1).astype(np.float)[:,None]
        ax.bar(test.Z, pi_emp_test[:,k], width=1, color='k')
        ax.plot(test.Z, test.pi[:,k], '-r', lw=2)
        lns[k,3] = ax.plot(test.Z, pi_test[:,k], '--r', lw=2)[0]

        # ax.set_xlim(0,)
        ax.set_ylim(0,1)
        ax.set_title("$\pi_%d$" % (k+1))

    plt.show()
    plt.pause(1.0)

    return lns

def update_plot(lns, model, train, test):

    plot_K = isinstance(model, LogisticNormalGP)

    # Make predictions at the training and testing data
    pi_train, psi_train, _ = \
            model.collapsed_predict(train.Z)

    pi_test, psi_test, _ = \
            model.collapsed_predict(test.Z)

    for k in range(K):
        if k == K-1 and not plot_K:
            pass
        else:
            lns[k,0].set_data(train.Z, psi_train[:,k])
            lns[k,1].set_data(test.Z, psi_test[:,k])

        lns[k,2].set_data(train.Z, pi_train[:,k])
        lns[k,3].set_data(test.Z, pi_test[:,k])

    plt.pause(0.001)

### Inference
Results = namedtuple("Results", ["lls", "pred_lls", "pred_pis", "pred_psis", "timestamps"])
def fit_model(model, train_data, test_data, N_iter=100, lns=None):

    if isinstance(model, EmpiricalStickBreakingGPModel):
        return fit_empirical_model(model, train_data, test_data)

    lls = [model.log_likelihood()]
    pred_lls = [model.predictive_log_likelihood(test_data.Z, test_data.X)[0]]
    pred_pi, pred_psi, _ = model.collapsed_predict(test_data.Z)
    pred_pis = [pred_pi]
    pred_psis = [pred_psi]

    timestamps = [time.clock()]
    for itr in range(N_iter):
        print("Iteration ", itr)
        model.resample_model()

        # Collect samples
        lls.append(model.log_likelihood())
        pred_lls.append(model.predictive_log_likelihood(test_data.Z, test_data.X)[0])
        pred_pi, pred_psi, _ = model.collapsed_predict(test_data.Z)
        pred_pis.append(pred_pi)
        pred_psis.append(pred_psi)
        timestamps.append(time.clock())

        # Update plots
        if lns is not None:
            update_plot(lns, model, train_data, test_data)

    # Compute sample mean and std
    lls = np.array(lls)
    pred_lls = np.array(pred_lls)
    pred_pis = np.array(pred_pis)
    pred_psis = np.array(pred_psis)
    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]

    return Results(lls, pred_lls, pred_pis, pred_psis, timestamps)


def fit_empirical_model(model, train, test):
    empirical_ll, _ = model.predictive_log_likelihood(train.Z, train.X)
    empirical_pred_ll, _ = model.predictive_log_likelihood(test.Z, test.X)
    pred_pi, pred_psi, _ = model.collapsed_predict(test.Z)
    pred_pis = np.array([pred_pi])
    pred_psis = np.array([pred_psi])
    # return Results(empirical_ll * np.ones(2),
    #                empirical_pred_ll * np.ones(2), [0, 1])
    return Results(empirical_ll * np.ones(2),
                   empirical_pred_ll * np.ones(2),
                   pred_pis, pred_psis, [0,1])

if __name__ == "__main__":

    train, test = initialize_test(N_max=N_max, true_model_class=MultinomialGP)

    # models = [EmpiricalStickBreakingGPModel, LogisticNormalGP, MultinomialGP]
    # labels = ["Emp GP", "LN GP", "LSB GP"]
    models = [EmpiricalStickBreakingGPModel, MultinomialGP]
    labels = ["Emp GP", "LSB GP"]

    results = []
    do_plot = False
    N_samples = 200
    for model_class in models:
        # Make a test model
        model = model_class(train.K, train.kernel, D=1)
        model.add_data(train.Z, train.X)

        # Initialize from the data
        model.initialize_from_data(initialize_to_mle=True)
        if isinstance(model, MultinomialGP):
            model.data_list[0]["psi"] = train.psi
            model.resample_omega()

        # Initialize plots
        if do_plot:
            lns = initialize_interactive_plot(model, train, test)
        else:
            lns = None

        # Inference
        res = fit_model(model, train, test,
                        N_iter=N_samples, lns=lns)
        results.append(res)

    fig, axs = plt.subplots(1,2)
    T_max = np.amax([np.amax(res.timestamps) for res in results])
    for ind, (res, label) in \
            enumerate(zip(results, labels)):
        axs[0].plot(res.timestamps, res.lls, color=colors[ind], label=label)
        axs[0].plot([0, T_max],
                 res.lls[len(res.lls)//2:].mean() * np.ones(2),
                 linestyle=":",
                 color=colors[ind])

        offset = len(res.pred_lls)//2
        axs[1].plot(res.timestamps, res.pred_lls, color=colors[ind], label=label)
        axs[1].plot([0, T_max],
                 res.pred_lls[offset:].mean() * np.ones(2),
                 linestyle=":",
                 color=colors[ind])

        # Plot the log-sum-exp of the pred_lls
        from scipy.misc import logsumexp
        expected_pred_ll =  logsumexp(res.pred_lls[offset:]) - np.log(len(res.pred_lls)-offset)
        axs[1].plot([0, T_max],
                    expected_pred_ll * np.ones(2),
                 linestyle="--",
                 color=colors[ind])

    axs[0].set_xlim(-1, T_max)
    axs[1].set_xlim(-1, T_max)
    plt.legend(loc="lower right")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Log likelihood")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Pred. Log likelihood")

    plt.ioff()
    plt.show()
