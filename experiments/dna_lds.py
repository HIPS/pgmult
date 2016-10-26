
import os
import gzip
import time
import pickle
import operator
import collections
import itertools
import numpy as np
from scipy.misc import logsumexp

import matplotlib.pyplot as plt
from hips.plotting.layout import create_axis_at_location, create_figure
import brewer2mpl

from pgmult.lds import MultinomialLDS
from pgmult.particle_lds import LogisticNormalMultinomialLDS, ParticleSBMultinomialLDS
from pgmult.hmm import MultinomialHMM
from pgmult.utils import pi_to_psi

from pylds.models import DefaultLDS

from pybasicbayes.distributions import GaussianFixed, Multinomial, Regression
from pybasicbayes.util.text import progprint_xrange
from autoregressive.distributions import AutoRegression

colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
colors = np.vstack([colors[:3], colors[4:5], colors[6:]])

np.seterr(invalid="warn")
np.random.seed(1)

#########################
#  set some parameters  #
#########################
stride = 1
min_subseq_len = 600
max_subseq_len = 1000

def discretize_seq(subseq, stride=1, key=None):
    subseq = subseq.lower()
    T = len(subseq) // stride
    # K = 4 ** stride
    X = np.zeros((T,100))

    if key is None:
        key = collections.defaultdict(itertools.count().__next__)

    for t in range(T):
        off = t * stride
        snip = subseq[off:off+stride]
        X[t, key[snip]] = 1

    # Check to see how many keys were actually used
    K = len(key)
    X = X[:,:K]

    return X, key

def load_data():
    ###################
    #  load dna data  #
    ###################
    from Bio import SeqIO
    import re
    import os
    # Download the data to data/dna/chr22.fa
    data_file = os.path.join("data", "dna", "chr22.fa")
    proc_data_file = os.path.join("data", "dna", "chr22_processed.pkl.gz")

    if os.path.exists(proc_data_file):
        with gzip.open(proc_data_file) as f:
            Xs,newkey = pickle.load(f)
    else:
        handle = open(data_file, "rU")
        for record in SeqIO.parse(handle, "fasta"):
            print(record.id)
        handle.close()

        # Parse this into a string and get rid of the N's
        # seq = str(record.seq)
        # interesting_subseqs = re.findall(r"[^N]+", seq)

        # Translate this into a protein sequence?
        # WARNING: This is computationally expensive!
        prot = record.seq.translate()
        prot = str(prot)
        interesting_subseqs = re.findall(r"[^X]+", prot)


        # Model all subsequences of length less than max_subseq_len
        Xs = []
        key = None
        subseq_lens = np.array([len(s) for s in interesting_subseqs])
        for subseq_ind, subseq in enumerate(interesting_subseqs):
            if len(subseq) > max_subseq_len or len(subseq) < min_subseq_len:
                continue
            print("Analyzing subsequence ", subseq_ind, " of length ", subseq_lens[subseq_ind])
            X, key = discretize_seq(subseq, stride=stride, key=key)
            assert (X.sum(axis=1) == 1).all()
            Xs.append(X)

        # Sort the Xs by usage
        usage = np.sum([X.sum(0) for X in Xs], axis=0)
        perm = np.argsort(usage)[::-1]
        for i,X in enumerate(Xs):
            Xs[i] = X[:,perm]

        # Update the key
        invkey = {v:k for k,v in list(key.items())}
        newkey = {k:v for k,v in list(key.items())}
        for i,j in enumerate(perm):
            newkey[invkey[j]] = i

        print(newkey)

        # Save the analyzed subsequences
        with gzip.open(proc_data_file, "w") as f:
            pickle.dump((Xs, dict(newkey)), f, protocol=-1)

    return Xs, newkey


# Inference stuff
# model, lls, test_lls, pred_lls, pis, psis, zs, timestamps
Results = collections.namedtuple("Results", ["lls", "test_lls", "pred_lls", "samples", "timestamps"])

def fit_lds_model(Xs, Xtest, D, N_samples=100):
    model = MultinomialLDS(K, D,
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=1*np.eye(D),M_0=np.zeros((D,D)),K_0=1*np.eye(D)),
        sigma_C=0.01
        )

    for X in Xs:
        model.add_data(X)

    model.resample_parameters()

    init_results = (0, model, model.log_likelihood(),
                    model.heldout_log_likelihood(Xtest, M=1),
                    model.predictive_log_likelihood(Xtest, M=1000))

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        return toc, None, model.log_likelihood(), \
            model.heldout_log_likelihood(Xtest, M=1), \
            model.predictive_log_likelihood(Xtest, M=1000)

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)

def fit_hmm(Xs, Xtest, D_hmm, N_samples=100):
    print("Fitting HMM with %d states" % D_hmm)
    model = MultinomialHMM(K, D_hmm)

    for X in Xs:
        model.add_data(X)

    init_results = (0, None, model.log_likelihood(),
                    model.log_likelihood(Xtest),
                    (model.log_likelihood(np.vstack((Xs[0], Xtest))) - model.log_likelihood(Xs[0])))

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        return toc, None, model.log_likelihood(), \
            model.log_likelihood(Xtest), \
            (model.log_likelihood(np.vstack((Xs[0], Xtest))) - model.log_likelihood(Xs[0]))

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)


def fit_gaussian_lds_model(Xs, Xtest, D_gauss_lds, N_samples=100):
    print("Fitting Gaussian (Raw) LDS with %d states" % D_gauss_lds)
    model = DefaultLDS(n=D_gauss_lds, p=K)

    Xs_centered = [X - np.mean(X, axis=0)[None,:] + 1e-3*np.random.randn(*X.shape) for X in Xs]
    for X in Xs_centered:
        model.add_data(X)

    # TODO: Get initial pred ll
    init_results = (0, None, np.nan, np.nan, np.nan)


    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        # Monte Carlo sample to get pi density implied by Gaussian LDS
        Tpred = Xtest.shape[0]
        Npred = 1000

        preds = model.sample_predictions(Xs_centered[0], Tpred, Npred=Npred)

        # Convert predictions to a distribution by finding the
        # largest dimension for each predicted Gaussian.
        # Preds is T x K x Npred, inds is TxNpred
        inds = np.argmax(preds, axis=1)
        pi = np.array([np.bincount(inds[t], minlength=K) for t in range(Tpred)]) / float(Npred)
        assert np.allclose(pi.sum(axis=1), 1.0)

        pi = np.clip(pi, 1e-8, 1.0)
        pi /= pi.sum(axis=1)[:,None]

        # Compute the log likelihood under pi
        pred_ll = np.sum([Multinomial(weights=pi[t], K=K).log_likelihood(Xtest[t][None,:])
                          for t in range(Tpred)])

        return toc, None, np.nan, \
            np.nan, \
            pred_ll

    n_retries = 0
    max_attempts = 5
    while n_retries < max_attempts:
        try:
            times, samples, lls, test_lls, pred_lls = \
                list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
            timestamps = np.cumsum(times)
            return Results(lls, test_lls, pred_lls, samples, timestamps)
        except Exception as e:
            print("Caught exception: ", e.message)
            print("Retrying")
            n_retries += 1

    raise Exception("Failed to fit the Raw Gaussian LDS model in %d attempts" % max_attempts)


def fit_ln_lds_model(Xs, Xtest, D, N_samples=100):
    """
    Fit a logistic normal LDS model with pMCMC
    """
    print("Fitting Logistic Normal LDS with %d states" % D)
    model = LogisticNormalMultinomialLDS(
        init_dynamics_distn=GaussianFixed(mu=np.zeros(D), sigma=1*np.eye(D)),
        dynamics_distn=AutoRegression(nu_0=D+1,S_0=D*np.eye(D),M_0=np.zeros((D,D)),K_0=D*np.eye(D)),
        emission_distn=Regression(nu_0=K+1,S_0=K*np.eye(K),M_0=np.zeros((K,D)),K_0=K*np.eye(D)),
        sigma_C=0.1)

    model.A = 0.5*np.eye(D)
    model.sigma_states = np.eye(D)
    model.C = 0.33 * np.random.randn(K,D)
    model.sigma_obs = 0.1*np.eye(K)

    for X in Xs:
        model.add_data(X)

    init_results = (0, None, model.log_likelihood(),
                    np.nan, model.predictive_log_likelihood(Xtest, Npred=1000))

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        pred_ll = model.predictive_log_likelihood(Xtest, Npred=1000)

        return toc, None, model.log_likelihood(), \
            np.nan, \
            pred_ll

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
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
        mu=pi_to_psi(np.ones(K)/K), sigma_C=0.01)

    model.A = 0.5*np.eye(D)
    model.sigma_states = np.eye(D)
    model.C = 0.01 * np.random.randn(K-1,D)
    model.sigma_obs = 0.1*np.eye(K)

    for X in Xs:
        model.add_data(X)

    init_results = (0, None, model.log_likelihood(),
                    np.nan, model.predictive_log_likelihood(Xtest, Npred=1000))

    def resample():
        tic = time.time()
        model.resample_model()
        toc = time.time() - tic

        pred_ll = model.predictive_log_likelihood(Xtest, Npred=1000)

        return toc, None, model.log_likelihood(), \
            np.nan, \
            pred_ll

    times, samples, lls, test_lls, pred_lls = \
        list(map(np.array, list(zip(*([init_results] + [resample() for _ in progprint_xrange(N_samples)])))))
    timestamps = np.cumsum(times)

    return Results(lls, test_lls, pred_lls, samples, timestamps)

# Plotting
def plot_qualitative_results(X, key, psi_lds, z_lds):
    start = 50
    stop = 70

    # Get the corresponding protein labels
    import operator
    id_to_char = dict([(v,k) for k,v in list(key.items())])
    sorted_chars = [idc[1].upper() for idc in sorted(list(id_to_char.items()), key=operator.itemgetter(0))]
    X_inds = np.where(X)[1]
    prot_str = [id_to_char[v].upper() for v in X_inds]


    from pgmult.utils import psi_to_pi
    pi_lds = psi_to_pi(psi_lds)

    # Plot the true and inferred states
    fig = create_figure(figsize=(3., 3.1))

    # Plot the string of protein labels
    # ax1 = create_axis_at_location(fig, 0.5, 2.5, 2.25, 0.25)
    # for n in xrange(start, stop):
    #     ax1.text(n, 0.5, prot_str[n].upper())
    # # ax1.get_xaxis().set_visible(False)
    # ax1.axis("off")
    # ax1.set_xlim([start-1,stop])
    # ax1.set_title("Protein Sequence")

    # ax2 = create_axis_at_location(fig, 0.5, 2.25, 2.25, 0.5)
    # ax2 = fig.add_subplot(311)
    # plt.imshow(X[start:stop,:].T, interpolation="none", vmin=0, vmax=1, cmap="Blues", aspect="auto")
    # ax2.set_title("One-hot Encoding")

    # ax3 = create_axis_at_location(fig, 0.5, 1.25, 2.25, 0.5)
    ax3 = fig.add_subplot(211)
    im3 = plt.imshow(np.kron(pi_lds[start:stop,:].T, np.ones((50,50))),
                             interpolation="none", vmin=0, vmax=1, cmap="Blues", aspect="auto",
               extent=(0,stop-start,K+1,1))
    # Circle true symbol
    from matplotlib.patches import Rectangle
    for n in range(start, stop):
        ax3.add_patch(Rectangle((n-start, X_inds[n]+1), 1, 1, facecolor="none", edgecolor="k"))

    # Print protein labels on y axis
    # ax3.set_yticks(np.arange(K))
    # ax3.set_yticklabels(sorted_chars)

    # Print protein sequence as xticks
    ax3.set_xticks(0.5+np.arange(0, stop-start))
    ax3.set_xticklabels(prot_str[start:stop])
    ax3.xaxis.tick_top()
    ax3.xaxis.set_tick_params(width=0)

    ax3.set_yticks(0.5+np.arange(1,K+1, 5))
    ax3.set_yticklabels(np.arange(1,K+1, 5))
    ax3.set_ylabel("$k$")

    ax3.set_title("Inferred Protein Probability", y=1.25)

    # Add a colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = plt.colorbar(im3, cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cbar.set_label("Probability", labelpad=10)


    # ax4 = create_axis_at_location(fig, 0.5, 0.5, 2.25, 0.55)
    lim = np.amax(abs(z_lds[start:stop]))
    ax4 = fig.add_subplot(212)
    im4 = plt.imshow(np.kron(z_lds[start:stop, :].T, np.ones((50,50))),
                     interpolation="none", vmin=-lim, vmax=lim, cmap="RdBu",
                     extent=(0,stop-start, D+1,1))
    ax4.set_xlabel("Position $t$")
    ax4.set_yticks(0.5+np.arange(1,D+1))
    ax4.set_yticklabels(np.arange(1,D+1))
    ax4.set_ylabel("$d$")

    ax4.set_title("Latent state sequence")

    # Add a colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar_ticks = np.round(np.linspace(-lim, lim, 3))
    cbar_ticks = [-4, 0, 4]
    cbar = plt.colorbar(im4, cax=cax,  ticks=cbar_ticks)
    # cbar.set_label("Probability", labelpad=10)


    # plt.subplots_adjust(top=0.9)
    # plt.tight_layout(pad=0.2)
    plt.savefig("dna_lds_1.png")
    plt.savefig("dna_lds_1.pdf")
    plt.show()

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
    plt.savefig(os.path.join("results", "dna", "run%03d" % run, outname))

    plt.tight_layout()

def plot_pred_log_likelihood(results, names, results_dir,
                             outname="pred_ll_vs_time.pdf",
                             smooth=True, burnin=2):
    # Plot the log likelihood
    fig = plt.figure(figsize=(2.5, 2.5))
    fig.set_tight_layout(True)
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

            plt.plot(np.clip(result.timestamps[burnin:], 1e-3,np.inf),
                     smooth_pred_ll[burnin:],
                     lw=2, color=colors[i], label=name)

        else:
            plt.plot(np.clip(result.timestamps[burnin:], 1e-3,np.inf),
                     result.pred_lls[burnin:],
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
    # plt.ylim(-700, -500)
    # plt.title("Chr22 DNA Model")
    plt.savefig(os.path.join(results_dir, outname))

def plot_pred_ll_vs_D(all_results, Ds, Xtrain, Xtest,
                      results_dir, models=None):
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
    normalizer = 0
    for Xtr, Xte in zip(Xtrain, Xtest):
        pi_emp = Xtr.sum(0) / float(Xtr.sum())
        pi_emp = np.clip(pi_emp, 1e-8, np.inf)
        pi_emp /= pi_emp.sum()
        baseline += Multinomial(weights=pi_emp, K=Xtr.shape[1]).log_likelihood(Xte).sum()
        normalizer += Xte.sum()

    # Make a bar chart with errorbars
    from hips.plotting.layout import create_figure
    fig = create_figure(figsize=(1.25,2.5), transparent=True)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    width = np.min(np.diff(Ds)) / (M+1.0) if len(Ds)>1 else 1.
    for m in range(M):
        ax.bar(Ds+m*width,
               (pred_ll_mean[:,m] - baseline) / normalizer,
               yerr=pred_ll_std[:,m] / normalizer,
               width=0.9*width, color=colors[m], ecolor='k')
        #
        # ax.text(Ds+(m-1)*width, yloc, rankStr, horizontalalignment=align,
        #     verticalalignment='center', color=clr, weight='bold')

    # Plot the zero line
    ax.plot([Ds.min()-width, Ds.max()+(M+1)*width], np.zeros(2), '-k')

    # Set the tick labels
    ax.set_xlim(Ds.min()-width, Ds.max()+(M+1)*width)
    # ax.set_xticks(Ds + (M*width)/2.)
    # ax.set_xticklabels(Ds)
    # ax.set_xticks(Ds + width * np.arange(M) + width/2. )
    # ax.set_xticklabels(models, rotation=45)
    ax.set_xticks([])

    # ax.set_xlabel("D")
    ax.set_ylabel("Pred. Log Lkhd. (nats/protein)")
    ax.set_title("DNA")

    plt.savefig(os.path.join(results_dir, "pred_ll_vs_D.pdf"))



def plot_figure_legend(results_dir):
    """
    Make a standalone legend
    :return:
    """
    from hips.plotting.layout import create_legend_figure
    labels = ["SBM-LDS (Gibbs)", "HMM (Gibbs)", "Raw LDS (Gibbs)", "LNM-LDS (pMCMC)"]
    fig = create_legend_figure(labels, colors[:4], size=(5.25,0.5),
                               lineargs={"lw": 2},
                               legendargs={"columnspacing": 0.75,
                                           "handletextpad": 0.1})
    fig.savefig(os.path.join(results_dir, "legend.pdf"))

if __name__ == "__main__":
    run = 5
    results_dir = os.path.join("results", "dna", "run%03d" % run)

    # Make sure the results directory exists
    from pgmult.utils import mkdir
    if not os.path.exists(results_dir):
        print("Making results directory: ", results_dir)
        mkdir(results_dir)

    # Load data
    Xs, key = load_data()

    # Split data into two
    T_end = Xs[0].shape[0]
    T_split = 10
    Xtrain = [Xs[0][:T_end-T_split,:]]
    Xtest = Xs[0][T_end-T_split:T_end,:]
    K = len(key)

    # Perform inference for a range of latent state dimensions and models
    N_samples = 1000
    all_results = []
    # Ds = np.array([2, 3, 4, 5, 6])
    Ds = np.array([4])
    models = ["SBM-LDS", "HMM", "Raw LDS" , "LNM-LDS"]
    methods = [fit_lds_model, fit_hmm, fit_gaussian_lds_model, fit_ln_lds_model]
    # Ds = np.array([2, 3, 4, 5, 6, 8, 10, 12, 14, 16])
    Ds = np.array([10])
    # models = ["SBM-LDS"]
    # methods = [fit_lds_model]

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

    # Plot log likelihoods for one value of D
    D_index = 0
    # plot_log_likelihood(all_results[D_index], models,
    #                     Sresults_dir,
    #                     outname="train_ll_vs_time_D%d.pdf" % Ds[D_index])

    # plot_pred_log_likelihood(all_results[D_index], models,
    #                      results_dir,
    #                     outname="pred_ll_vs_time_D%d.pdf" % D_index)
    #
    # # Make a bar chart of all the results
    plot_pred_ll_vs_D(all_results, Ds, Xtrain, [Xtest], results_dir)
    plt.show()

    # lds_model = all_results[0][0].samples[0]
    # lds_z = lds_model.data_list[0]["states"].stateseq
    # lds_psi = lds_z.dot(lds_model.C.T) + lds_model.emission_distn.mu
    # plot_qualitative_results(Xs[0], key, lds_psi, lds_z)

    # plot_figure_legend(results_dir)
