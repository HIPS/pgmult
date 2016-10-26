
import os
import numpy as np
import pickle
from hips.plotting.layout import create_figure
from hips.plotting.colormaps import gradient_cmap
import matplotlib.pyplot as plt
import brewer2mpl
from scipy.misc import logsumexp

from spclust import find_blockifying_perm

colors = brewer2mpl.get_map("Set1", "Qualitative",  9).mpl_colors
goodcolors = np.array([0,1,2,4,6,7,8])
colors = np.array(colors)[goodcolors]

def logma(v):
    def logavg(v):
        return logsumexp(v) - np.log(len(v))

    return np.array([logavg(v[n//2:n]) for n in range(2,len(v))])

def corr_matrix(a):
     return a / np.outer(np.sqrt(np.diag(a)), np.sqrt(np.diag(a)))

def top_k(words, pi, k=8):
        # Get the top k words ranked by pi
        perm = np.argsort(pi)[::-1]
        return np.array(words)[perm][:k]

def plot_correlation_matrix(Sigma,
                            betas,
                            words,
                            results_dir,
                            outname="corr_matrix.pdf",
                            blockify=False,
                            highlight=[]):

    # Get topic names
    topic_names = [np.array(words)[np.argmax(beta)]  for beta in betas.T]

    # Plot the log likelihood
    sz = 5.25/3.  # Three NIPS panels
    fig = create_figure(figsize=(sz, 2.5), transparent=True)
    fig.set_tight_layout(True)
    ax = fig.add_subplot(111)

    C = corr_matrix(Sigma)
    T = C.shape[0]
    lim = abs(C).max()
    cmap = gradient_cmap([colors[1], np.ones(3), colors[0]])

    if blockify:
        perm = find_blockifying_perm(C, k=4, nclusters=4)
        C = C[np.ix_(perm, perm)]

    im = plt.imshow(np.kron(C, np.ones((50,50))), interpolation="none", vmin=-lim, vmax=lim, cmap=cmap, extent=(1,T+1,T+1,1))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax,
                        orientation="horizontal",
                        ticks=[-1, -0.5, 0., 0.5, 1.0],
                        label="Topic Correlation")
    # cbar.set_label("Probability", labelpad=10)
    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.9, right=0.85)

    # Highlight some cells
    import string
    from matplotlib.patches import Rectangle
    for i,(j,k) in enumerate(highlight):
        ax.add_patch(Rectangle((k+1, j+1), 1, 1, facecolor="none", edgecolor='k', linewidth=1))

        ax.text(k+1-1.5,j+1+1,string.ascii_lowercase[i], )

        print("")
        print("CC: ", C[j,k])
        print("Topic ", j)
        print(top_k(words, betas[:,j]))
        print("Topic ", k)
        print(top_k(words, betas[:,k]))
        print("")

    # Find the most correlated off diagonal entry
    C_offdiag = np.tril(C,k=-1)
    sorted_pairs = np.argsort(C_offdiag.ravel())
    for i in range(5):
        print("")
        imax,jmax = np.unravel_index(sorted_pairs[-i], (T,T))
        print("Correlated Topics (%d, %d): " % (imax, jmax))
        print(top_k(words, betas[:,imax]), "\n and \n", top_k(words, betas[:,jmax]))
        print("correlation coeff: ", C[imax, jmax])
        print("-" * 50)
        print("")

    print("-" * 50)
    print("-" * 50)
    print("-" * 50)

    for i in range(5):
        print("")
        imin,jmin = np.unravel_index(sorted_pairs[i], (T,T))
        print("Anticorrelated Topics (%d, %d): " % (imin, jmin))
        # print topic_names[imin], " and ", topic_names[jmin]
        print(top_k(words, betas[:,imin]), "\n and \n", top_k(words, betas[:,jmin]))
        print("correlation coeff: ", C[imin, jmin])
        print("-" * 50)
        print("")


    # Move main axis ticks to top
    ax.xaxis.tick_top()
    # ax.set_title("Topic Correlation", y=1.1)
    fig.savefig(os.path.join(results_dir, outname))

    plt.show()

def print_topics(betas, words):
    for t, beta in enumerate(betas.T):
        print("Topic ", t)
        print(top_k(words, beta, k=10))
        print("")


def plot_pred_log_likelihood(timestamp_list,
                             pred_ll_list, names,
                             results_dir,
                             outname="ctm_pred_ll_vs_time.pdf",
                             title=None,
                             smooth=True, burnin=3,
                             normalizer=4632.       # Number of words in test dataset
                            ):
    # Plot the log likelihood
    width = 5.25/3.  # Three NIPS panels
    fig = create_figure(figsize=(width, 2.25), transparent=True)
    fig.set_tight_layout(True)

    min_time = np.min([np.min(times[burnin+2:]) for times in timestamp_list])
    max_time = np.max([np.max(times[burnin+2:]) for times in timestamp_list])

    for i,(times, pred_ll, name) in enumerate(zip(timestamp_list, pred_ll_list, names)[::-1]):

        # Smooth the log likelihood
        smooth_pred_ll = logma(pred_ll)
        plt.plot(np.clip(times[burnin+2:], 1e-3,np.inf),
                 smooth_pred_ll[burnin:] / normalizer,
                 lw=2, color=colors[3-i], label=name)

        # plt.plot(np.clip(times[burnin:], 1e-3,np.inf),
        #          pred_ll[burnin:] / normalizer,
        #          lw=2, color=colors[3-i], label=None)

        N = len(pred_ll)
        avg_pll = logsumexp(pred_ll[N//2:]) - np.log(N-N//2)

        plt.plot([min_time, max_time], avg_pll / normalizer * np.ones(2), ls='--', color=colors[3-i])


    plt.xlabel('Time [s] (log scale) ', fontsize=9)
    plt.xscale("log")
    plt.xlim(min_time, max_time)

    plt.ylabel("Pred. Log Lkhd. [nats/word]", fontsize=9)
    plt.ylim(-2.6, -2.47)
    plt.yticks([-2.6, -2.55, -2.5])
    # plt.subplots_adjust(left=0.05)

    if title:
        plt.title(title)

    # plt.ylim(-9.,-8.4)
    plt.savefig(os.path.join(results_dir, outname))
    plt.show()


def plot_figure_legend(results_dir):
    """
    Make a standalone legend
    :return:
    """
    from hips.plotting.layout import create_legend_figure
    labels = ["SBM-CTM (Gibbs)", "LNM-CTM (Gibbs)", "CTM (EM)", "LDA (Gibbs)"]
    fig = create_legend_figure(labels, colors[:4], size=(5.25,0.5),
                               lineargs={"lw": 2},
                               legendargs={"columnspacing": 0.75,
                                           "handletextpad": 0.125})
    fig.savefig(os.path.join(results_dir, "ctm_legend.pdf"))

if __name__ == "__main__":
    # res_dir = os.path.join("results", "newsgroups_lda")
    # res_file = os.path.join(res_dir, "lda_bignewsgroups_results2.pkl")
    res_dir = os.path.join("results", "ap_lda")
    res_file = os.path.join(res_dir, "ec2_results2_c.pkl")
    with open(res_file) as f:
        res = pickle.load(f)

    # Plot pred ll vs time
    models = ["sb", "ln", "em", "lda"]
    names = ["SBM-CTM (Gibbs)", "LNM-CTM (Gibbs)", "CTM (EM)", "LDA (collapsed Gibbs)"]
    # models = ["sb", "em"]
    # names = ["SBM-CTM", "LNM-CTM (EM)"]
    times = [np.array(res[k]["timestamps"]) for k in models]
    pred_lls = [np.array(res[k]["predictive_lls"]) for k in models]
    plot_pred_log_likelihood(times, pred_lls, names, res_dir, title="AP News")


    # Plot the correlation matrix
    # from experiments.newsgroups_lda import load_ap_data, load_newsgroup_data
    # from pgmult.lda import StickbreakingCorrelatedLDA
    # counts, words = load_newsgroup_data(V=1000, cats=None)
    # last_sample = res["sb"]["samples"]
    # assert isinstance(last_sample, StickbreakingCorrelatedLDA)
    #
    # # Topics to highlight:
    # # highlight = [(9,6), (3,10)]
    # highlight = []

    # plot_correlation_matrix(last_sample.theta_prior.sigma,
    #                         last_sample.beta,
    #                         words,
    #                         res_dir,
    #                         highlight=highlight)
    #
    # from pgmult.utils import psi_to_pi
    # topic_probs = psi_to_pi(last_sample.theta_prior.mu)
    # topic_perm = np.argsort(topic_probs)
    # print_topics(last_sample.beta[:,topic_perm], words)

    plot_figure_legend(res_dir)
