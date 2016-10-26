
import os, re
import numpy as np
import scipy
from scipy.misc import logsumexp
from scipy.special import gammaln, beta
from scipy.integrate import simps
from numpy import newaxis as na

import scipy.sparse
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix


import pypolyagamma as ppg

from pgmult.internals.dirichlet import log_dirichlet_density


def initialize_polya_gamma_samplers():
    if "OMP_NUM_THREADS" in os.environ:
        num_threads = int(os.environ["OMP_NUM_THREADS"])
    else:
        num_threads = ppg.get_omp_num_threads()
    assert num_threads > 0

    # Choose random seeds
    seeds = np.random.randint(2**16, size=num_threads)
    return [ppg.PyPolyaGamma(seed) for seed in seeds]


def initialize_pyrngs():
    from gslrandom import PyRNG, get_omp_num_threads
    if "OMP_NUM_THREADS" in os.environ:
        num_threads = os.environ["OMP_NUM_THREADS"]
    else:
        num_threads = get_omp_num_threads()
    assert num_threads > 0

    # Choose random seeds
    seeds = np.random.randint(2**16, size=num_threads)
    return [PyRNG(seed) for seed in seeds]

def log_polya_gamma_density(x, b, c, trunc=1000):
    if np.isscalar(x):
        xx = np.array([[x]])
    else:
        assert x.ndim == 1
        xx = x[:,None]

    logf = np.zeros(xx.size)
    logf += b * np.log(np.cosh(c/2.))
    logf += (b-1) * np.log(2) - gammaln(b)

    # Compute the terms in the summation
    ns = np.arange(trunc)[None,:].astype(np.float)
    terms = np.zeros_like(ns, dtype=np.float)
    terms += gammaln(ns+b) - gammaln(ns+1)
    terms += np.log(2*ns+b) - 0.5 * np.log(2*np.pi)

    # Compute the terms that depend on x
    terms = terms - 3./2*np.log(xx)
    terms += -(2*ns+b)**2 / (8*xx)
    terms += -c**2/2. * xx

    # logf += logsumexp(terms, axis=1)

    maxlogf = np.amax(terms, axis=1)[:,None]
    logf += np.log(np.exp(terms - maxlogf).dot((-1.0)**ns.T)).ravel() \
            + maxlogf.ravel()
    # terms2 = terms.reshape((xx.shape[0], -1, 2))
    # df = -np.diff(np.exp(terms2 - terms2.max(2)[...,None]),axis=2)
    # terms2 = np.log(df+0j) + terms2.max(2)[...,None]
    # logf += logsumexp(terms2.reshape((xx.shape[0], -1)), axis=1)

    # plt.figure()
    # plt.plot(xx, logf)

    return logf

def polya_gamma_density(x, b, c, trunc=1000):
    return np.exp(log_polya_gamma_density(x, b, c, trunc)).real

def logistic(x):
    return 1./(1+np.exp(-x))

def logit(p):
    return np.log(p/(1-p))

def psi_to_pi(psi, axis=None):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector
    :return:        Length K normalized probability vector
    """
    if axis is None:
        if psi.ndim == 1:
            K = psi.size + 1
            pi = np.zeros(K)

            # Set pi[1..K-1]
            stick = 1.0
            for k in range(K-1):
                pi[k] = logistic(psi[k]) * stick
                stick -= pi[k]

            # Set the last output
            pi[-1] = stick
            # DEBUG
            assert np.allclose(pi.sum(), 1.0)

        elif psi.ndim == 2:
            M, Km1 = psi.shape
            K = Km1 + 1
            pi = np.zeros((M,K))

            # Set pi[1..K-1]
            stick = np.ones(M)
            for k in range(K-1):
                pi[:,k] = logistic(psi[:,k]) * stick
                stick -= pi[:,k]

            # Set the last output
            pi[:,-1] = stick

            # DEBUG
            assert np.allclose(pi.sum(axis=1), 1.0)

        else:
            raise ValueError("psi must be 1 or 2D")
    else:
        K = psi.shape[axis] + 1
        pi = np.zeros([psi.shape[dim] if dim != axis else K for dim in range(psi.ndim)])
        stick = np.ones(psi.shape[:axis] + psi.shape[axis+1:])
        for k in range(K-1):
            inds = [slice(None) if dim != axis else k for dim in range(psi.ndim)]
            pi[inds] = logistic(psi[inds]) * stick
            stick -= pi[inds]
        pi[[slice(None) if dim != axis else -1 for dim in range(psi.ndim)]] = stick
        assert np.allclose(pi.sum(axis=axis), 1.)

    return pi

def pi_to_psi(pi):
    """
    Convert probability vector pi to a vector psi
    :param pi:      Length K probability vector
    :return:        Length K-1 transformed vector psi
    """
    if pi.ndim == 1:
        K = pi.size
        assert np.allclose(pi.sum(), 1.0)
        psi = np.zeros(K-1)

        stick = 1.0
        for k in range(K-1):
            psi[k] = logit(pi[k] / stick)
            stick -= pi[k]

        # DEBUG
        assert np.allclose(stick, pi[-1])
    elif pi.ndim == 2:
        M, K = pi.shape
        assert np.allclose(pi.sum(axis=1), 1.0)
        psi = np.zeros((M,K-1))

        stick = np.ones(M)
        for k in range(K-1):
            psi[:,k] = logit(pi[:,k] / stick)
            stick -= pi[:,k]
        assert np.allclose(stick, pi[:,-1])
    else:
        raise NotImplementedError

    return psi

def det_jacobian_pi_to_psi(pi):
    """
    Compute |J| = |d\psi_j / d\pi_k| = the jacobian of the mapping from
     pi to psi. Since we have a stick breaking construction, the Jacobian
     is lower triangular and the determinant is simply the product of the
     diagonal. For our model, this can be computed in closed form. See the
     appendix of the draft.

    :param pi: K dimensional probability vector
    :return:
    """
    # import pdb; pdb.set_trace()
    K = pi.size

    # Jacobian is K-1 x K-1
    diag = np.zeros(K-1)
    for k in range(K-1):
        diag[k] = (1.0 - pi[:k].sum()) / (pi[k] * (1-pi[:(k+1)].sum()))

    det_jacobian = diag.prod()
    return det_jacobian

def det_jacobian_psi_to_pi(psi):
    """
    Compute the Jacobian of the inverse mapping psi to pi.
    :param psi:
    :return:
    """
    pi = psi_to_pi(psi)
    return 1.0 / det_jacobian_pi_to_psi(pi)


def dirichlet_to_psi_density(pi_mesh, alpha):
    psi_mesh = np.array(list(map(pi_to_psi, pi_mesh)))
    valid_psi = np.all(np.isfinite(psi_mesh), axis=1)
    psi_mesh = psi_mesh[valid_psi,:]

    # Compute the det of the Jacobian of the inverse mapping
    det_jacobian = 1.0 / np.array(list(map(det_jacobian_pi_to_psi, pi_mesh)))
    det_jacobian = det_jacobian[valid_psi]

    # Compute the Dirichlet density
    pi_pdf = np.exp(log_dirichlet_density(pi_mesh, alpha=alpha))
    pi_pdf = pi_pdf[valid_psi]

    # The psi density is scaled by the det of the Jacobian
    psi_pdf = pi_pdf * det_jacobian

    return psi_mesh, psi_pdf

def dirichlet_to_psi_density_closed_form(pi_mesh, alpha):
    psi_mesh = np.array(list(map(pi_to_psi, pi_mesh)))
    valid_psi = np.all(np.isfinite(psi_mesh), axis=1)
    psi_mesh = psi_mesh[valid_psi,:]

    # import ipdb; ipdb.set_trace()
    Z = np.exp(gammaln(alpha.sum()) - gammaln(alpha).sum())
    sigma_psi = logistic(psi_mesh)
    sigma_negpsi = logistic(-psi_mesh)
    alpha_sum = np.cumsum(alpha[::-1])[::-1][1:]

    # alphasum should = [\sum_{j=2}^K \alpha_j, ..., \alpha_{K-1} + \alpha_K, \alpha_K]
    psi_pdf = sigma_psi**alpha[None, :-1] * sigma_negpsi**alpha_sum[None, :]
    psi_pdf = Z * psi_pdf.prod(axis=1)

    return psi_mesh, psi_pdf

def gaussian_to_pi_density(psi_mesh, mu, Sigma):
    pi_mesh = np.array(list(map(psi_to_pi, psi_mesh)))
    valid_pi = np.all(np.isfinite(pi_mesh), axis=1)
    pi_mesh = pi_mesh[valid_pi,:]

    # Compute the det of the Jacobian of the inverse mapping
    det_jacobian = np.array(list(map(det_jacobian_pi_to_psi, pi_mesh)))
    det_jacobian = det_jacobian[valid_pi]

    # Compute the multivariate Gaussian density
    # pi_pdf = np.exp(log_dirichlet_density(pi_mesh, alpha=alpha))
    from scipy.stats import multivariate_normal
    psi_dist = multivariate_normal(mu, Sigma)
    psi_pdf = psi_dist.pdf(psi_mesh)
    psi_pdf = psi_pdf[valid_pi]

    # The psi density is scaled by the det of the Jacobian
    pi_pdf = psi_pdf * det_jacobian

    return pi_mesh, pi_pdf

def ln_psi_to_pi(psi):
    """
    Convert the logistic normal psi to a probability vector pi
    :param psi:     Length K vector
    :return:        Length K normalized probability vector
    """
    lognumer = psi

    if psi.ndim == 1:
        logdenom = logsumexp(psi)
    elif psi.ndim == 2:
        logdenom = logsumexp(psi, axis=1)[:, None]
    pi = np.exp(lognumer - logdenom)
    # assert np.allclose(pi.sum(), 1.0)

    return pi

def ln_pi_to_psi(pi, scale=1.0):
    """
    Convert the logistic normal psi to a probability vector pi
    The transformation from psi to pi is not invertible unless
    you know the scaling of the psis.

    :param pi:      Length K vector
    :return:        Length K unnormalized real vector
    """
    assert scale > 0
    if pi.ndim == 1:
        assert np.allclose(pi.sum(), 1.0)
    elif pi.ndim == 2:
        assert np.allclose(pi.sum(1), 1.0)

    psi = np.log(pi) + np.log(scale)

    # assert np.allclose(pi, ln_psi_to_pi(psi))
    return psi

def compute_uniform_mean_psi(K, alpha=2):
    """
    Compute the multivariate distribution over psi that will yield approximately
    Dirichlet(\alpha) prior over pi

    :param K:   Number of entries in pi
    :return:    A K-1 vector mu that yields approximately uniform distribution over pi
    """
    mu, sigma = compute_psi_cmoments(alpha*np.ones(K))
    return mu, np.diag(sigma)

def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    mu = np.zeros(K-1)
    sigma = np.zeros(K-1)
    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        mu[k] = simps(psi*density(psi),psi)
        sigma[k] = simps(psi**2*density(psi),psi) - mu[k]**2
        # print '%d: mean=%0.3f var=%0.3f' % (k, mean, s - mean**2)

    return mu, sigma

def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / scipy.special.beta(alpha_k,alpha_rest)
    return density

def plot_psi_marginals(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    import matplotlib.pyplot as plt
    plt.figure()

    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        plt.subplot(2,1,1)
        plt.plot(psi,density(psi),label='psi_%d' % k)
        plt.subplot(2,1,2)
        plt.plot(psi,np.log(density(psi)),label='psi_%d' % k)
    plt.subplot(2,1,1)
    plt.legend()

def N_vec(x, axis=None):
    """
    Compute the count vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            N = x.sum()
            return np.concatenate(([N], N - np.cumsum(x)[:-2]))
        elif x.ndim == 2:
            N = x.sum(axis=1)
            return np.hstack((N[:,None], N[:,None] - np.cumsum(x, axis=1)[:,:-2]))
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else None for dim in range(x.ndim)]
        inds2 = [slice(None) if dim != axis else slice(None,-2) for dim in range(x.ndim)]
        N = x.sum(axis=axis)
        return np.concatenate((N[inds], N[inds] - np.cumsum(x,axis=axis)[inds2]), axis=axis)

def kappa_vec(x, axis=None):
    """
    Compute the kappa vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            return x[:-1] - N_vec(x)/2.0
        elif x.ndim == 2:
            return x[:,:-1] - N_vec(x)/2.0
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else slice(None,-1) for dim in range(x.ndim)]
        return x[inds] - N_vec(x, axis)/2.0

# is this doing overlapping work with dirichlet_to_psi_density_closed_form?
def get_marginal_psi_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / beta(alpha_k,alpha_rest)
    return density


def dirichlet_to_psi_meanvar(alphas,psigrid=np.linspace(-10,10,1000)):
    K = alphas.shape[0]

    def meanvar(k):
        density = get_marginal_psi_density(alphas[k], alphas[k+1:].sum())
        mean = simps(psigrid*density(psigrid),psigrid)
        s = simps(psigrid**2*density(psigrid),psigrid)
        return mean, s - mean**2

    return list(map(np.array, list(zip(*[meanvar(k) for k in range(K-1)]))))


def cumsum(v,strict=False):
    if not strict:
        return np.cumsum(v,axis=0)
    else:
        out = np.zeros_like(v)
        out[1:] = np.cumsum(v[:-1],axis=0)
        return out

def list_split(lst,num):
    assert 0 < num <= len(lst)
    lens = [len(lst[start::num]) for start in range(num)]
    starts, stops = cumsum(lens,strict=True), cumsum(lens,strict=False)
    return [lst[start:stop] for start,stop in zip(starts,stops)]

def flatten(lst):
    return [item for sublist in lst for item in sublist]


def plot_gaussian_2D(mu, Sigma, color='b',centermarker=True,label='',alpha=1.,ax=None,artists=None):
    from matplotlib import pyplot as plt
    ax = ax if ax else plt.gca()

    t = np.hstack([np.arange(0,2*np.pi,0.01),0])
    circle = np.vstack([np.sin(t),np.cos(t)])
    ellipse = np.dot(np.linalg.cholesky(Sigma),circle)

    if artists is None:
        point = ax.scatter([mu[0]],[mu[1]],marker='D',color=color,s=4,alpha=alpha) \
            if centermarker else None
        line, = ax.plot(
            ellipse[0,:] + mu[0], ellipse[1,:] + mu[1],linestyle='-',
            linewidth=2,color=color,label=label,alpha=alpha)
    else:
        line = artists[0]
        if centermarker:
            point = artists[1]
            point.set_offsets(np.atleast_2d(mu))
            point.set_alpha(alpha)
            point.set_color(color)
        else:
            point = None
        line.set_xdata(ellipse[0,:] + mu[0])
        line.set_ydata(ellipse[1,:] + mu[1])
        line.set_alpha(alpha)
        line.set_color(color)

    return (line, point) if point else (line,)

def solve_diagonal_plus_lowrank(diag_of_A,B,C,b):
    '''
    like np.linalg.solve(np.diag(diag_of_A)+B.dot(C),b) but better!
    b can be a matrix
    see p.673 of Convex Optimization by Boyd and Vandenberghe
    '''
    # TODO write a psd version where B=C.T
    one_dim = b.ndim == 1
    if one_dim:
        b = np.reshape(b,(-1,1))
    z = b/diag_of_A[:,na]
    E = C.dot(B/diag_of_A[:,na])
    E.flat[::E.shape[0]+1] += 1
    w = np.linalg.solve(E,C.dot(z))
    z -= B.dot(w)/diag_of_A[:,na]
    return z if not one_dim else z.ravel()


def mkdir(path):
    # http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def downsample_data_slow(X, n):
    """
    Downsample each row of X such that it sums to n by randomly removing entries
    """
    from pybasicbayes.util.stats import sample_discrete
    assert X.ndim == 2

    Xsub = X.copy()

    for i in range(Xsub.shape[0]):
        Mi = int(Xsub[i].sum())
        assert Mi >= n
        # if Mi > 1e8: print "Warning: M is really large!"
        p = Xsub[i] / float(Mi)

        # Random remove one of the entries to remove
        for m in range(Mi-n):
            k = sample_discrete(p)
            assert Xsub[i,k] > 0
            Xsub[i,k] -= 1
            p = Xsub[i] / float(Xsub[i].sum())

        assert Xsub[i].sum() == n

    return Xsub


def downsample_data(X, n):
    """
    Downsample each row of X such that it sums to n by randomly removing entries
    """
    from pybasicbayes.util.general import ibincount
    assert X.ndim == 2
    D,K = X.shape

    Xsub = X.copy().astype(np.int)

    for d in range(D):
        xi = ibincount(Xsub[d])
        Xsub[d] = np.bincount(np.random.choice(xi, size=n, replace=False), minlength=K)

        assert Xsub[d].sum() == n

    return Xsub.astype(np.float)


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

    print(('loaded {} documents with a size {} vocabulary'.format(*counts.shape)))
    print(('with {} words per document on average'.format(np.mean(counts.sum(1)))))
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
