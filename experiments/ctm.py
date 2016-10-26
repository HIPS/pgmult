"""
Correlated topic model (LDA) test
"""


import os, re, gzip, time, operator, inspect, hashlib, random
import pickle as pickle
from collections import namedtuple
from functools import wraps

import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.misc import logsumexp

import numpy as np
np.random.seed(11223344)

import matplotlib.pyplot as plt
import brewer2mpl
colors = brewer2mpl.get_map("Set1", "Qualitative", 9).mpl_colors
colors = colors[:3] + [colors[4]]

from pybasicbayes.util.text import progprint_xrange

from pgmult.lda import StandardLDA, StickbreakingCorrelatedLDA, LogisticNormalCorrelatedLDA
from pgmult.utils import mkdir, ln_psi_to_pi, pi_to_psi

all_categories = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'rec.autos', 'rec.motorcycles',
    'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
    'sci.med', 'sci.space', 'misc.forsale', 'talk.politics.misc',
    'talk.politics.guns', 'talk.politics.mideast', 'talk.religion.misc',
    'alt.atheism', 'soc.religion.christian',
]

cachedir = 'result_cache'

# If you want to compare against the CTM of Blei and Lafferty,
# set this to true and rerun. If you haven't installed it to
# deps/ctm-dist, the code will halt and print an error message
# with a link to where the CTM code can be found.
FIT_BLEI_CTM = False

# If you want to compare against the logistic normal CTM with
# Polya-gamma augmentation, set this to true. Note that this
# fitting procedure is much slower
FIT_LN_CTM = False

# If you want to compare against standard collapsed LDA, set
# this to true. This takes about 10 mins on my Intel Core i7.
FIT_LDA = True

# To fit our stick breaking CTM, set this to true. This takes
# about 30 mins on my Intel Core i7.
FIT_SB_CTM = True

##########
#  util  #
##########

def cached(func):
    mkdir(cachedir)
    cachebase = os.path.join(cachedir, func.__module__ + func.__name__)

    def replace_arrays(v):
        if isinstance(v, np.ndarray):
            return hashlib.sha1(v).hexdigest()
        if isinstance(v, scipy.sparse.csr.csr_matrix):
            out = hashlib.sha1(v.data)
            out.update(v.indices)
            out.update(v.indptr)
            return out.hexdigest()
        return v

    @wraps(func)
    def wrapped(*args, **kwargs):
        argdict = \
            {k:replace_arrays(v) for k,v in
                inspect.getcallargs(func,*args,**kwargs).items()}
        closurevals = \
            [replace_arrays(cell.cell_contents) for cell in func.__closure__ or []]

        key = str(hash(frozenset(list(argdict.items()) + closurevals)))
        cachefile = cachebase + '.' + key

        if os.path.isfile(cachefile):
            with gzip.open(cachefile, 'r') as infile:
                value = pickle.load(infile)
            return value
        else:
            value = func(*args,**kwargs)
            with gzip.open(cachefile, 'w') as outfile:
                pickle.dump(value, outfile, protocol=-1)
            return value

    return wrapped


#############
#  loading  #
#############

def load_newsgroup_data(V, cats, sort_data=True):
    from sklearn.datasets import fetch_20newsgroups
    print("Downloading newsgroups data...")
    print('cats = %s' % cats)
    newsgroups = fetch_20newsgroups(
        subset="train", categories=cats, remove=('headers', 'footers', 'quotes'))
    return get_sparse_repr(newsgroups.data, V, sort_data)


def load_ap_data(V, sort_data=True):
    def fetch_ap():
        from io import BytesIO
        from urllib.request import urlopen
        import tarfile

        print("Downloading AP data...")
        response = urlopen('http://www.cs.princeton.edu/~blei/lda-c/ap.tgz')
        tar = tarfile.open(fileobj=BytesIO(response.read()))
        return str(tar.extractfile('ap/ap.txt').read())

    ap = fetch_ap().replace("\\n", '')
    docs = re.findall(r'(?<=<TEXT> )(.*?)(?= </TEXT>)', ap)
    doclen = [len(d) for d in docs]
    print("Number of Documents: ", len(docs))
    print("Average Document Length: ", np.mean(doclen))
    return get_sparse_repr(docs, V, sort_data)


def sample_documents(counts, words, num=10):
    for row in random.sample(list(counts), num):
        for col in row.nonzero()[1]:
            print('{} '.format(words[col]))
        print('\n')


def split_test_train(data, train_frac, test_frac, exclude_words_not_in_training=True):
    '''
    Split the sparse matrix 'data' into two sparse matrices, 'train_data', and
    'test_data', such that data = train_data + test_data.

    Each row in data corresponds to a document and each column corresponds to a
    word in the vocabulary, so that data is shape (num_docs, vocab_size) and
    data[i,j] = count of word j in document i.

    The test data held out from the training data is a subset of words in a
    subset of documents. That is, the data matrix is first split into two types
    of row: training rows and testing rows. Training rows are used in training
    without any of their data held out. For testing rows, the aim is to train on
    a subset of the words in the document and test how well the remaining words
    can be predicted. Therefore this function performs two splits: first to
    split into training rows and testing rows, and second to split the testing
    rows into training counts and held-out counts.

    The 'train_frac' and 'test_frac' inputs control what fraction of rows are
    marked as training rows and what fraction of words within test rows are held
    out, respectively.
    '''

    def split_rows(mat, p):
        def sparse_to_lilrows(mat):
            out = mat.tolil()
            return list(zip(out.rows, out.data))

        def lilrows_to_csr(lilrows):
            out = lil_matrix(mat.shape, dtype=np.uint32)
            out.rows, out.data = list(zip(*lilrows))
            return out.tocsr()

        def split(lst, p):
            def _split(lst, inds):
                return [x if ind else ([], []) for x, ind in zip(lst, inds)]
            inds = np.random.rand(len(lst)) < p
            return _split(lst, ~inds), _split(lst, inds)

        return list(map(lilrows_to_csr, split(sparse_to_lilrows(mat), p)))

    def split_multinomial(mat, p):
        data2 = np.random.binomial(mat.data, p)
        data1 = mat.data - data2
        m1 = csr_matrix((data1, mat.indices, mat.indptr), mat.shape, dtype=np.uint32)
        m2 = csr_matrix((data2, mat.indices, mat.indptr), mat.shape, dtype=np.uint32)
        return m1, m2

    # split out training rows and testing rows
    test_data, train_data = split_rows(data, train_frac)

    # within testing rows, hold out some fraction of the words
    test_train, test_data = split_multinomial(test_data, test_frac)

    # for the counts within testing rows that aren't held out, add them back to
    # the training data
    train_data += test_train

    assert train_data.sum() + test_data.sum() == data.sum()

    if exclude_words_not_in_training:
        unseen = np.asarray((test_data.sum(0) > 0) & (train_data.sum(0) == 0)).ravel()
        print('{} test words were never seen in training data'.format(unseen.sum()))
        test_data.data[np.in1d(test_data.indices, np.where(unseen)[0])] = 0
        assert np.sum((test_data.sum(0) > 0) & (train_data.sum(0) == 0)) == 0

    train_data.eliminate_zeros()
    test_data.eliminate_zeros()

    return train_data, test_data


def get_sparse_repr(docs, V, sort_data):
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(stop_words="english", max_features=V)
    default_preproc = vectorizer.build_preprocessor()

    def preproc(s):
        return re.sub(r' \d+ ', 'anumber ', default_preproc(s))

    vectorizer.preprocessor = preproc

    counts = vectorizer.fit_transform(docs).astype(np.uint32)
    words = vectorizer.get_feature_names()
    if sort_data:
        counts, words = sort_vocab(counts, words)
        assert is_column_sorted(counts)

    print('loaded {} documents with a size {} vocabulary'.format(*counts.shape))
    print('with {} words per document on average'.format(np.mean(counts.sum(1))))
    print()

    return counts, words


def sort_vocab(counts, words):
    tots = counts.T.dot(np.ones(counts.shape[0]))
    words = [words[idx] for idx in np.argsort(-tots)]
    counts = sort_columns_by_counts(counts)
    return counts, words


def sparse_from_blocks(blocks):
    blocklen = lambda data_indices: data_indices[0].shape[0]
    data, indices = list(map(np.concatenate, list(zip(*blocks))))
    indptr = np.concatenate(((0,), np.cumsum(list(map(blocklen, blocks)))))
    return data, indices, indptr


def sparse_to_blocks(mat):
    data, indices, indptr = mat.data, mat.indices, mat.indptr
    slices = list(map(slice, indptr[:-1], indptr[1:]))
    return [(data[sl], indices[sl]) for sl in slices]


def sort_columns_by_counts(mat):
    count = lambda data_indices1: data_indices1[0].sum()
    sorted_cols = sorted(sparse_to_blocks(mat.tocsc()), key=count, reverse=True)
    return csc_matrix(sparse_from_blocks(sorted_cols), mat.shape).tocsr()


def is_column_sorted(mat):
    a = np.asarray(mat.sum(0)).ravel()
    return np.all(a == a[np.argsort(-a)])


#############
#  fitting  #
#############

Results = namedtuple(
    "Results",
    ["loglikes", "predictive_lls", "perplexities", "samples", "timestamps"])


def fit_lnctm_em(train_data, test_data, T):
    import pgmult.internals.ctm_wrapper as ctm_wrapper
    if ctm_wrapper.has_ctm_c:
        print('Running CTM EM...')
        return Results(*ctm_wrapper.fit_ctm_em(train_data, test_data, T))


def sampler_fitter(name, cls, method, initializer):
    def fit(train_data, test_data, T, Niter, init_at_em, *args):
        resample = operator.methodcaller(method)

        def evaluate(model):
            ll, pll, perp = \
                model.log_likelihood(), model.log_likelihood(test_data), \
                model.perplexity(test_data)
            return ll, pll, perp

        def sample(model):
            tic = time.time()
            resample(model)
            timestep = time.time() - tic
            return evaluate(model), timestep

        print('Running %s...' % name)
        model = cls(train_data, T, *args)
        model = initializer(model) if init_at_em and initializer else model
        init_val = evaluate(model)
        vals, timesteps = list(zip(*[sample(model) for _ in progprint_xrange(Niter)]))

        lls, plls, perps = list(zip(*((init_val,) + vals)))
        timestamps = np.cumsum((0.,) + timesteps)

        return Results(lls, plls, perps, model.copy_sample(), timestamps)

    fit.__name__ = name
    return fit


def make_ctm_initializer(get_psi):
    if FIT_BLEI_CTM:
        import pgmult.internals.ctm_wrapper as ctm_wrapper
        ctm_initial_beta_path = os.path.join(ctm_wrapper.resultsdir, '000-log-beta.dat')
        ctm_initial_lambda_path = os.path.join(ctm_wrapper.resultsdir, '000-lambda.dat')

        def initializer(model):
            T, V = model.T, model.V

            model.beta = np.exp(np.loadtxt(ctm_initial_beta_path)
                                .reshape((-1,V))).T

            lmbda = np.loadtxt(ctm_initial_lambda_path).reshape((-1,T))
            nonempty_docs = np.asarray(model.data.sum(1) > 0).ravel()
            model.psi[nonempty_docs] = get_psi(lmbda)

            model.resample_z()
            model.resample_theta_prior()

            return model

    else:
        def initializer(model):
            return model

    return initializer


def lda_initializer(model):
    if FIT_BLEI_CTM:
        T, V = model.T, model.V
        model.beta = np.exp(np.loadtxt('ctm-out/000-log-beta.dat')
                                .reshape((-1,V))).T
        lmbda = np.loadtxt('ctm-out/000-lambda.dat').reshape((-1,T))
        nonempty_docs = np.asarray(model.data.sum(1) > 0).ravel()
        model.theta[nonempty_docs] = ln_psi_to_pi(lmbda)
        model.resample_z()
    return model


fit_lda_gibbs = sampler_fitter(
    'fit_lda_gibbs', StandardLDA, 'resample', lda_initializer)
fit_lda_collapsed = sampler_fitter(
    'fit_lda_collapsed', StandardLDA, 'resample_collapsed', lda_initializer)
fit_lnctm_gibbs = sampler_fitter(
    'fit_lnctm_gibbs', LogisticNormalCorrelatedLDA, 'resample',
    make_ctm_initializer(lambda lmbda: lmbda))
fit_sbctm_gibbs = sampler_fitter(
    'fit_sbctm_gibbs', StickbreakingCorrelatedLDA, 'resample',
    make_ctm_initializer(lambda lmbda: pi_to_psi(ln_psi_to_pi(lmbda))))


########################
#  inspecting results  #
########################

def plot_sb_interpretable_results(sb_results, words):
    nwords = 5
    Sigma = sb_results[-1][-1]
    T = Sigma.shape[0]

    def get_topwords(topic):
        return words[np.argsort(sb_results[-1][0][:,topic])[-nwords:]]

    lim = np.abs(Sigma).max()
    plt.imshow(np.kron(Sigma,np.ones((50,50))), extent=(0,T,T,0), vmin=-lim, vmax=lim,
               cmap='RdBl')
    plt.colorbar()

    for t in range(T):
        print('Topic %d:' % t)
        print(get_topwords(t))
        print()


def print_topics(std_results, std_collapsed_results, sb_results, ln_results, words, T):
    words = np.array(words)
    nwords = 5

    def get_topwords(result, topic):
        return words[np.argsort(result.samples[-1][0][:,topic])[-nwords:]]

    names = ['Standard LDA', 'Collapsed LDA', 'LN Correlated LDA', 'SB Correlated LDA']
    results = [std_results, std_collapsed_results, sb_results, ln_results]

    for name, result in zip(names, results):
        print('Top words for %s' % name)
        for t in range(T):
            print('Topic {}: {}'.format(t, ' '.join(get_topwords(result, t))))
        print()


def plot_figure_legend():
    """
    Make a standalone legend
    :return:
    """
    from hips.plotting.layout import create_legend_figure
    labels = ["SB-CTM", "LN-CTM (Gibbs)", "LN-CTM (VB)", "LDA (Gibbs)"]
    fig = create_legend_figure(labels, colors[:4], size=(5.25,0.5),
                               lineargs={"lw": 2},
                               legendargs={"columnspacing": 0.75,
                                           "handletextpad": 0.125})
    fig.savefig(os.path.join("results", "newsgroups", "legend.pdf"))


def logma(v):
    def logavg(v):
        return logsumexp(v) - np.log(len(v))

    return np.array([logavg(v[n//2:n]) for n in range(2,len(v))])


def plot_predictive_lls(result, logaddexp, **kwargs):
    plot_kwargs = dict(lw=2, marker='o', markersize=2, markeredgecolor='none')
    plot_kwargs.update(kwargs)
    if 'color' in kwargs:
        plot_kwargs['markerfacecolor'] = kwargs['color']

    plt.plot(result.timestamps, result.predictive_lls, **plot_kwargs)
    if logaddexp:
        plt.plot(result.timestamps[2:], logma(result.predictive_lls), **plot_kwargs)

    print(result.predictive_lls[-10:])

    plt.legend(loc='lower right')
    plt.xlabel('Time (sec)')
    plt.ylabel('Held-out predictive log likelihoods')
    plt.tight_layout()


#########################
#  running experiments  #
#########################

if __name__ == '__main__':
    ## newsgroups
    # cats = None  # all categories
    # T, V = 50, 1000
    # alpha_beta, alpha_theta = 0.1, 1.
    # train_frac, test_frac = 0.9, 0.5
    # data, words = load_newsgroup_data(V, cats)

    ## AP
    T, V = 100, 1000
    alpha_beta, alpha_theta = 0.05, 1.
    train_frac, test_frac = 0.95, 0.5
    data, words = load_ap_data(V)

    ## print setup
    print('T=%d, V=%d' % (T, V))
    print('alpha_beta = %0.3f, alpha_theta = %0.3f' % (alpha_beta, alpha_theta))
    print('train_frac = %0.3f, test_frac = %0.3f' % (train_frac, test_frac))
    print()

    ## split train test
    train_data, test_data = split_test_train(data, train_frac=train_frac, test_frac=test_frac)

    ## fit and plot
    all_results = dict()
    if FIT_BLEI_CTM:
        em_results = fit_lnctm_em(train_data, test_data, T)
        if em_results is not None:
            plot_predictive_lls(em_results, False, color=colors[0], label='LN CTM EM')

        all_results['em'] = em_results

    if FIT_LN_CTM:
        ln_results = fit_lnctm_gibbs(train_data, test_data, T, 200, True, alpha_beta)
        plot_predictive_lls(ln_results, True, color=colors[1], label='LN CTM Gibbs')
        all_results['ln'] = ln_results

    if FIT_LDA:
        lda_results = fit_lda_collapsed(train_data, test_data, T, 1000, True, alpha_beta, alpha_theta)
        plot_predictive_lls(lda_results, True, color=colors[1], label='LDA Gibbs')
        all_results['lda'] = lda_results

    if FIT_SB_CTM:
        sb_results = fit_sbctm_gibbs(train_data, test_data, T, 1500, True, alpha_beta)
        plot_predictive_lls(sb_results, True, color=colors[2], label='SB CTM Gibbs')
        all_results['sb'] = sb_results

    with open('ctm_results.pkl','w') as outfile:
        pickle.dump(all_results, outfile, protocol=-1)

    plt.show()
