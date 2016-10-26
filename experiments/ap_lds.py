"""
Linear dynamical system model for the AP text dataset.
Each document is modeled as a draw from an LDS with
categorical observations.
"""

import os
import re
import gzip
import time
import pickle
import operator
import collections
import numpy as np
from scipy.misc import logsumexp
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
from hips.plotting.layout import create_axis_at_location, create_figure
import brewer2mpl

from pgmult.lds import MultinomialLDS
from pgmult.particle_lds import LogisticNormalMultinomialLDS, ParticleSBMultinomialLDS
from pgmult.hmm import MultinomialHMM
from pgmult.utils import pi_to_psi

from pylds.models import DefaultLDS, NonstationaryLDS

from pybasicbayes.distributions import GaussianFixed, Multinomial, Regression
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
goodcolors = np.array([0,1,2,4,6,7,8])
colors = np.array(colors)[goodcolors]

np.seterr(invalid="warn")
np.random.seed(0)

np.seterr(invalid="warn")
np.random.seed(0)

# Model parameters
K = 200     # Number of words

# Data handling
def load(filename=None):
    if filename is None:
        bigstr = download_ap()
    else:
        with open(filename,'r') as infile:
            bigstr = infile.read()

    docs = re.findall(r'(?<=<TEXT> ).*?(?= </TEXT>)',bigstr.translate(None,'\n'))

    vectorizer = CountVectorizer(stop_words='english',max_features=K).fit(docs)
    docs = [make_onehot_seq(doc, vectorizer) for doc in docs]

    words = vectorizer.get_feature_names()

    return docs, words

def download_ap():
    from io import StringIO
    from urllib.request import urlopen
    import tarfile

    print("Downloading AP data...")
    response = urlopen('http://www.cs.princeton.edu/~blei/lda-c/ap.tgz')
    tar = tarfile.open(fileobj=StringIO(response.read()))
    return tar.extractfile('ap/ap.txt').read()

def filter_wordseq(doc, vectorizer):
    return [w for w in doc if w in vectorizer.vocabulary_]

def make_onehot_seq(doc, vectorizer):
    lst = filter_wordseq(vectorizer.build_analyzer()(doc), vectorizer)
    indices = {word:idx for idx, word in enumerate(vectorizer.vocabulary_.keys())}

    out = np.zeros((len(lst),len(indices)))
    for wordidx, word in enumerate(lst):
        out[wordidx, indices[word]] = 1
    return out


# Inference stuff
# model, lls, test_lls, pred_lls, pis, psis, zs, timestamps
Results = collections.namedtuple("Results", ["lls", "test_lls", "pred_lls", "samples", "timestamps"])

def fit_lds_model(Xs, Xtest, D, N_samples=100):
    Nx = len(Xs)
    assert len(Xtest) == Nx

    model = MultinomialLDS(K, D,
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=1*np.eye(D),M_0=np.zeros((D,D)),K_0=1*np.eye(D)),
        sigma_C=1.
        )

    for X in Xs:
        model.add_data(X)

    model.resample_parameters()

    compute_pred_ll = lambda: sum([model.predictive_log_likelihood(Xt, data_index=i, M=10)[0]
                                   for i,Xt in enumerate(Xtest)])

    init_results = (0, None, model.log_likelihood(),
                    # model.heldout_log_likelihood(Xtest, M=1),
                    np.nan,
                    compute_pred_ll())

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        return toc, None, model.log_likelihood(), \
            np.nan,\
            compute_pred_ll()

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] +
            [resample() for _ in progprint_xrange(N_samples, perline=5)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)

def fit_hmm(Xs, Xtest, D_hmm, N_samples=100):
    print("Fitting HMM with %d states" % D_hmm)
    model = MultinomialHMM(K, D_hmm, alpha_0=10.0)

    for X in Xs:
        model.add_data(X)

    compute_pred_ll = lambda: sum([model.log_likelihood(np.vstack((Xs[i], Xtest[i])))
                                   - model.log_likelihood(Xs[i])
                                   for i,Xt in enumerate(Xtest)])

    init_results = (0, None, model.log_likelihood(),
                    model.log_likelihood(Xtest),
                    compute_pred_ll())

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        return toc, None, model.log_likelihood(), \
            np.nan, \
            compute_pred_ll()

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] +
            [resample() for _ in progprint_xrange(N_samples, perline=5)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)


def fit_gaussian_lds_model(Xs, Xtest, D_gauss_lds, N_samples=100):
    print("Fitting Gaussian (Raw) LDS with %d states" % D_gauss_lds)
    # model = DefaultLDS(n=D_gauss_lds, p=K)
    # model = DefaultLDS(n=D_gauss_lds, p=K)
    from pylds.models import NonstationaryLDS
    model = NonstationaryLDS(
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=1*np.eye(D),M_0=np.zeros((D,D)),K_0=1*np.eye(D)),
        emission_distn=Regression(nu_0=K+1,S_0=K*np.eye(K),M_0=np.zeros((K,D)),K_0=K*np.eye(D)))

    Xs_centered = [X - np.mean(X, axis=0)[None,:] + 1e-3*np.random.randn(*X.shape) for X in Xs]
    for X in Xs_centered:
        model.add_data(X)

    def compute_pred_ll():
        pred_ll = 0
        for i,Xt in enumerate(Xtest):
            # Monte Carlo sample to get pi density implied by Gaussian LDS
            Npred = 10
            Tpred = Xt.shape[0]
            preds = model.sample_predictions(Xs_centered[i], Tpred, Npred=Npred)

            # Convert predictions to a distribution by finding the
            # largest dimension for each predicted Gaussian.
            # Preds is T x K x Npred, inds is TxNpred
            inds = np.argmax(preds, axis=1)
            pi = np.array([np.bincount(inds[t], minlength=K) for t in range(Tpred)]) / float(Npred)
            assert np.allclose(pi.sum(axis=1), 1.0)

            pi = np.clip(pi, 1e-8, 1.0)
            pi /= pi.sum(axis=1)[:,None]

            # Compute the log likelihood under pi
            pred_ll += np.sum([Multinomial(weights=pi[t], K=K).log_likelihood(Xt[t][None,:])
                              for t in range(Tpred)])

        return pred_ll

    # TODO: Get initial pred ll
    init_results = (0, None, np.nan, np.nan, compute_pred_ll())

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic


        return toc, None, np.nan, \
            np.nan, \
            compute_pred_ll()


    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] +
            [resample() for _ in progprint_xrange(N_samples, perline=5)])))))
    timestamps = np.cumsum(times)
    return Results(lls, test_lls, pred_lls, samples, timestamps)


def fit_ln_lds_model(Xs, Xtest, D, N_samples=100):
    """
    Fit a logistic normal LDS model with pMCMC
    """
    print("Fitting Logistic Normal LDS with %d states" % D)
    model = LogisticNormalMultinomialLDS(
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=D*np.eye(D),M_0=np.zeros((D,D)),K_0=D*np.eye(D)),
        emission_distn=Regression(nu_0=K+1,S_0=K*np.eye(K),M_0=np.zeros((K,D)),K_0=K*np.eye(D)))

    model.A = 0.5*np.eye(D)
    model.sigma_states = np.eye(D)
    model.C = 0.33*np.random.randn(K,D)
    model.sigma_obs = 0.1*np.eye(K)

    for X in Xs:
        model.add_data(X)

    compute_pred_ll = lambda: sum([model.predictive_log_likelihood(Xt, data_index=i, Npred=10)[0]
                                   for i,Xt in enumerate(Xtest)])

    init_results = (0, None, model.log_likelihood(),
                    np.nan, compute_pred_ll())

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        # pred_ll = model.predictive_log_likelihood(Xtest, Npred=1000)

        return toc, None, model.log_likelihood(), \
            np.nan, \
            compute_pred_ll()

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] +
            [resample() for _ in progprint_xrange(N_samples, perline=5)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)


def fit_lds_model_with_pmcmc(Xs, Xtest, D, N_samples=100):
    """
    Fit a logistic normal LDS model with pMCMC
    """
    print("Fitting SBM-LDS with %d states using pMCMC" % D)
    model = ParticleSBMultinomialLDS(
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=D*np.eye(D),M_0=np.zeros((D,D)),K_0=D*np.eye(D)),
        emission_distn=Regression(nu_0=K+1,S_0=K*np.eye(K),M_0=np.zeros((K,D)),K_0=K*np.eye(D)),
        mu=pi_to_psi(np.ones(K)/K))

    model.A = 0.5*np.eye(D)
    model.sigma_states = np.eye(D)
    model.C = np.random.randn(K-1,D)
    model.sigma_obs = 0.1*np.eye(K)

    for X in Xs:
        model.add_data(X)

    compute_pred_ll = lambda: sum([model.predictive_log_likelihood(Xt, data_index=i, Npred=10)[0]
                                   for i,Xt in enumerate(Xtest)])

    init_results = (0, None, model.log_likelihood(),
                    np.nan, compute_pred_ll())

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        # pred_ll = model.predictive_log_likelihood(Xtest, Npred=1000)

        return toc, None, model.log_likelihood(), \
            np.nan, \
            compute_pred_ll()

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)

def plot_log_likelihood(results, names, run, outname="pred_ll_vs_time.pdf"):
    # Plot the log likelihood
    plt.figure(figsize=(3,3.2))
    for i,(result, name) in enumerate(zip(results, names)):
        plt.plot(result.timestamps, result.lls, lw=2, color=colors[i], label=name)

    # plt.plot(gauss_lds_lls, lw=2, color=colors[2], label="Gaussian LDS")
    plt.legend(loc="lower right")
    plt.xlabel('Time (s)')
    plt.ylabel("Log Likelihood")
    # plt.title("Chr22 DNA Model")
    plt.savefig(os.path.join("results", "ap", "run%03d" % run, outname))

    plt.tight_layout()

def plot_pred_log_likelihood(results, names, run, outname="pred_ll_vs_time.pdf", smooth=True):
    # Plot the log likelihood
    plt.figure(figsize=(3,3.2))
    for i,(result, name) in enumerate(zip(results, names)):
        if result.pred_lls.ndim == 2:
            pred_ll = result.pred_lls[:,0]
        else:
            pred_ll = result.pred_lls


        # Smooth the log likelihood
        if smooth:
            win = 10
            pad_pred_ll = np.concatenate((pred_ll[0] * np.ones(win), pred_ll))
            smooth_pred_ll = np.array([logsumexp(pad_pred_ll[j-win:j+1])-np.log(win)
                                       for j in range(win, pad_pred_ll.size)])

            plt.plot(np.clip(result.timestamps, 1e-3,np.inf), smooth_pred_ll,
                     lw=2, color=colors[i], label=name)

        else:
            plt.plot(np.clip(result.timestamps, 1e-3,np.inf), result.pred_lls,
                     lw=2, color=colors[i], label=name)


        # if result.pred_lls.ndim == 2:
        #     plt.errorbar(np.clip(result.timestamps, 1e-3,np.inf),
        #                  result.pred_lls[:,0],
        #                  yerr=result.pred_lls[:,1],
        #                  lw=2, color=colors[i], label=name)
        # else:
        #     plt.plot(np.clip(result.timestamps, 1e-3,np.inf), result.pred_lls, lw=2, color=colors[i], label=name)


    # plt.plot(gauss_lds_lls, lw=2, color=colors[2], label="Gaussian LDS")
    # plt.legend(loc="lower right")
    plt.xlabel('Time (s)')
    plt.xscale("log")
    plt.ylabel("Pred. Log Likelihood")
    plt.ylim(-700, -500)
    # plt.title("Chr22 DNA Model")
    plt.savefig(os.path.join("results", "ap", "run%03d" %run, outname))

    plt.tight_layout()

def plot_pred_ll_vs_D(all_results, Ds, Xtrain, Xtest, run):
    # Create a big matrix of shape (len(Ds) x 5 x T) for the pred lls
    N = len(Ds)                             # Number of dimensions tests
    M = len(all_results[0])                 # Number of models tested
    T = len(all_results[0][0].pred_lls)     # Number of MCMC iters
    pred_lls = np.zeros((N,M,T))
    for n in range(N):
        for m in range(M):
            if all_results[n][m].pred_lls.ndim == 2:
                pred_lls[n,m] = all_results[n][m].pred_lls[:,0]
            else:
                pred_lls[n,m] = all_results[n][m].pred_lls

    # Compute the mean and standard deviation on burned in samples
    burnin = T // 2
    pred_ll_mean = logsumexp(pred_lls[:,:,burnin:], axis=-1) - np.log(T-burnin)

    # Use bootstrap to compute error bars
    pred_ll_std = np.zeros_like(pred_ll_mean)
    for n in range(N):
        for m in range(M):
            samples = np.random.choice(pred_lls[n,m,burnin:], size=(100, (T-burnin)), replace=True)
            pll_samples = logsumexp(samples, axis=1) - np.log(T-burnin)
            pred_ll_std[n,m] = pll_samples.std()

    # Get the baseline pred ll
    baseline = 0
    for Xtr, Xte in zip(Xtrain, Xtest):
        pi_emp = Xtr.sum(0) / float(Xtr.sum())
        pi_emp = np.clip(pi_emp, 1e-8, np.inf)
        pi_emp /= pi_emp.sum()
        baseline += Multinomial(weights=pi_emp, K=Xtr.shape[1]).log_likelihood(Xte).sum()

    # Make a bar chart with errorbars
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    width = np.min(np.diff(Ds)) / (M+1.0) if len(Ds)>1 else 1.
    for m in range(M):
        ax.bar(Ds+m*width,
               pred_ll_mean[:,m] - baseline,
               yerr=pred_ll_std[:,m],
               width=0.9*width, color=colors[m], ecolor='k')

    # Plot the zero line
    ax.plot([Ds.min()-width, Ds.max()+(M+1)*width], np.zeros(2), '-k')

    # Set the tick labels
    ax.set_xlim(Ds.min()-width, Ds.max()+(M+1)*width)
    ax.set_xticks(Ds + (M*width)/2.)
    ax.set_xticklabels(Ds)

    ax.set_xlabel("D")
    ax.set_ylabel("Pred. Log Likelihood")
    ax.set_title("AP News")

    plt.savefig(os.path.join("results", "ap", "run%03d" % run, "pred_ll_vs_D.pdf"))

def fit_joint_corpus():
    run = 13
    results_dir = os.path.join("results", "ap", "run%03d" % run)

    # Make sure the results directory exists
    from pgmult.internals.utils import mkdir
    if not os.path.exists(results_dir):
        print("Making results directory: ", results_dir)
        mkdir(results_dir)

    # Load the AP news documents
    Xs, words = load()

    N_docs = 10
    T_split = 10

    # Filter out documents shorter than 2 * T_split
    Xfilt = [X for X in Xs if X.shape[0] > 2*T_split]
    Xtrain = [X[:-T_split] for X in Xfilt[:N_docs]]
    Xtest = [X[-T_split:] for X in Xfilt[:N_docs]]

    # Perform inference for a range of latent state dimensions and models
    N_samples = 500
    all_results = []
    Ds = np.array([10])
    models = ["SBM-LDS", "HMM", "Raw LDS" , "LNM-LDS", "SBM-LDS (pMCMC)"]
    methods = [fit_lds_model, fit_hmm, fit_gaussian_lds_model, fit_ln_lds_model, fit_lds_model_with_pmcmc]

    for D in Ds:
        D_results = []
        for model, method in zip(models, methods):
            results_file = os.path.join(results_dir, "results_%s_D%d.pkl.gz" % (model, D))
            if os.path.exists(results_file):
                print("Loading from: ", results_file)
                with gzip.open(results_file, "r") as f:
                    D_model_results = pickle.load(f)
            else:
                print("Fitting ", model, " for D=",D)
                D_model_results = method(Xtrain, Xtest, D, N_samples)

                with gzip.open(results_file, "w") as f:
                    print("Saving to: ", results_file)
                    pickle.dump(D_model_results, f, protocol=-1)

            D_results.append(D_model_results)
        all_results.append(D_results)

    # Plot log likelihoods for the results using one D
    res_index = 0
    plot_log_likelihood(all_results[res_index],
                        models,
                        run,
                        outname="train_ll_vs_time_D%d.pdf" % Ds[res_index])

    plot_pred_log_likelihood(all_results[res_index],
                        models,
                        run,
                        outname="pred_ll_vs_time_D%d.pdf" % Ds[res_index])

    # Make a bar chart of all the results
    plot_pred_ll_vs_D(all_results, Ds, Xtrain, Xtest, run)
    plt.show()


if __name__ == "__main__":
    fit_joint_corpus()
