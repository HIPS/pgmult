"""
Correlated LDA test
"""
import time
import operator
import functools
from collections import namedtuple

import numpy as np
np.random.seed(11223344)
import matplotlib.pyplot as plt
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors

from pybasicbayes.util.text import progprint_xrange
from pybasicbayes.util.general import ibincount
from pgmult.lda import StandardLDA, StickbreakingCorrelatedLDA, LogisticNormalCorrelatedLDA

Results = namedtuple("Results", ["lls", "perplexity", "samples", "timestamps"])
def train_model(model, train_data, test_data, N_samples=300, method='resample_model', thetas=None):
    print 'Training %s with %s' % (model.__class__.__name__, method)
    model.add_data(train_data)

    # Initialize to a given set of thetas
    if thetas is not None:
        model.thetas = thetas
        for d in model.documents:
            d.resample_z()

    init_like, init_perp, init_sample, init_time = \
        model.log_likelihood(), model.perplexity(test_data), \
        model.copy_sample(), time.time()

    def update(i):
        operator.methodcaller(method)(model)
        # print "ll: ", model.log_likelihood()
        return model.log_likelihood(), \
               model.perplexity(test_data), \
               model.copy_sample(), \
               time.time()

    likes, perps, samples, timestamps = zip(*[update(i) for i in progprint_xrange(N_samples,perline=5)])

    # Get relative timestamps
    timestamps = np.array((init_time,) + timestamps)
    timestamps -= timestamps[0]

    return Results((init_like,) + likes,
                   (init_perp,) + perps,
                   (init_sample,) + samples,
                   timestamps)

def generate_synth_data(V=10, D=10, T=5, N=100, alpha_beta=10., alpha_theta=10., plot=False, train_frac=0.5):
    # true_lda = StandardLDA(T, V, alpha_beta=alpha_beta, alpha_theta=alpha_theta)
    true_lda = LogisticNormalCorrelatedLDA(T, V, alpha_beta=alpha_beta)

    print "Sigma: ", true_lda.Sigma

    # true_lda = StickbreakingCorrelatedLDA(T, V, alpha_beta=alpha_beta)
    data = np.zeros((D,V),dtype=int)
    for d in xrange(D):
        doc = true_lda.generate(N=N, keep=True)
        data[d,:] = doc.w

    if plot:
        plt.figure()
        plt.imshow(data, interpolation="none")
        plt.xlabel("Vocabulary")
        plt.ylabel("Documents")
        plt.colorbar()
        plt.show()

    # Split each document into two
    train_data = np.zeros_like(data)
    test_data = np.zeros_like(data)
    for d,w in enumerate(data):
        # Get vector where i is repeated w[i] times
        wcnt = ibincount(w)

        # Subsample wcnt
        train_inds = np.random.rand(wcnt.size) < train_frac
        train_data[d] = np.bincount(wcnt[train_inds], minlength=V)
        test_data[d]  = np.bincount(wcnt[~train_inds], minlength=V)

        assert np.allclose(train_data[d] + test_data[d], w)

    return true_lda, train_data, test_data


if __name__ == '__main__':
    # generate synthetic data
    alpha_beta, alpha_theta = 2., 2.

    # Small N (hard for LDA)
    D = 200
    V = 1000
    N = 50
    T = 5
    N_samples = 100

    # Big N (easy for standard LDA)
    # D = 100
    # V = 50
    # N = 1000
    # T = 8
    # N_samples = 100


    true_lda, train_data, test_data = generate_synth_data(V, D, T, N, alpha_beta, alpha_theta)
    print 'Generating with D=%d, V=%d, N=%d, T=%d, a_beta=%0.3f, a_theta=%0.3f' % \
        (D,V,N,T,alpha_beta,alpha_theta)
    print

    # train models with MCMC
    train = functools.partial(train_model, train_data=train_data, test_data=test_data, N_samples=N_samples)

    ## DEBUG! Set beta and theta to true values
    init_to_true = False
    std_model = StandardLDA(T,V,alpha_beta,alpha_theta)
    std_model.beta = true_lda.beta if init_to_true else std_model.beta
    std_results = \
        train(std_model, thetas=true_lda.thetas if init_to_true else None)

    std_collapsed_model = StandardLDA(T,V,alpha_beta,alpha_theta)
    std_collapsed_model.beta = true_lda.beta if init_to_true else std_collapsed_model.beta
    std_collapsed_results = \
        train(std_collapsed_model,
              method='resample_model_collapsed',
              thetas=true_lda.thetas if init_to_true else None)

    sb_model = StickbreakingCorrelatedLDA(T, V, alpha_beta)
    sb_model.beta = true_lda.beta if init_to_true else sb_model.beta
    sb_results = \
        train(sb_model, thetas=true_lda.thetas if init_to_true else None)

    ln_model = LogisticNormalCorrelatedLDA(T, V, alpha_beta)
    ln_model.beta = true_lda.beta if init_to_true else ln_model.beta
    ln_results = \
        train(ln_model, thetas=true_lda.thetas if init_to_true else None)

    all_results = [sb_results, ln_results, std_results, std_collapsed_results]
    all_labels = ["SB Corr. LDA", "LN Corr. LDA", "Std. LDA", "Collapsed LDA"]
    # all_results = [std_results, std_collapsed_results]
    # all_labels = ["Std. LDA", "Collapsed LDA"]
    # all_results = [ln_results]
    # all_labels = ["LN Corr. LDA"]

    plt.figure()
    # Plot log likelihood vs iteration
    plt.subplot(121)
    for ind, (results, label) in enumerate(zip(all_results, all_labels)):
        plt.plot(results.timestamps, results.lls,
             color=colors[ind], lw=2,
             marker="o", markersize=4, markerfacecolor=colors[ind], markeredgecolor="none",
             label=label)

    # Plot the true log likelihood
    T_max = np.amax([res.timestamps[-1] for res in all_results])
    plt.plot([0, T_max], true_lda.heldout_log_likelihood(train_data) * np.ones(2), 'k:')
    plt.xlim(0, T_max)

    plt.legend(loc="lower right")
    plt.xlabel("Time (s)")
    plt.ylabel("Training Log Lkhd")
    plt.title("LDA: D=%d, V=%d, T=%d, N=%d" % (D,V,T,N))

    # Plot test perplexity vs iteration
    plt.subplot(122)
    for ind, (results, label) in enumerate(zip(all_results, all_labels)):
        plt.plot(results.timestamps, results.perplexity,
             color=colors[ind], lw=2,
             marker="o", markersize=4, markerfacecolor=colors[ind], markeredgecolor="none",
             label=label)

    # Plot the true perplexity
    plt.plot([0, T_max], true_lda.perplexity(test_data) * np.ones(2), 'k:')
    plt.xlim(0, T_max)

    plt.legend(loc="upper right")
    plt.xlabel("Time (s)")
    plt.ylabel("Perplexity")
    plt.title("LDA: D=%d, V=%d, T=%d, N=%d" % (D,V,T,N))

    plt.show()

    # from pgmult.utils.profiling import show_line_stats
    # show_line_stats()
