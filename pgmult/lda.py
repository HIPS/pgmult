"""
Latent Dirichlet Allocation models, including vanilla LDA as well as correlated
and dynamic topic models (CTMs and DTMs) employing the Polya-Gamma augmentation.
"""
import abc
import copy
import numpy as np

from scipy.misc import logsumexp
from scipy.linalg import solve_triangular as _solve_triangular
from scipy.linalg.lapack import dpotrs as _dpotrs
from scipy.special import gammaln
import scipy.sparse

from pypolyagamma import pgdrawvpar
from pybasicbayes.distributions import Gaussian

from gslrandom import multinomial_par

from pgmult.internals.utils import \
    kappa_vec, N_vec, \
    compute_uniform_mean_psi, psi_to_pi, pi_to_psi, \
    ln_pi_to_psi, ln_psi_to_pi, \
    initialize_polya_gamma_samplers, \
    initialize_pyrngs


###
# Util
###

def dpotrs(L, a):
    return _dpotrs(L, a, lower=True)[0]


def solve_triangular(L, a):
    return _solve_triangular(L, a, lower=True, trans='T')


def sample_dirichlet(a, normalize):
    if 'vertical'.startswith(normalize):
        return np.hstack([np.random.dirichlet(col)[:,None] for col in a.T])
    else:
        return np.vstack([np.random.dirichlet(row) for row in a])


def csr_nonzero(mat):
    rows = np.arange(mat.shape[0]).repeat(np.diff(mat.indptr))
    cols = mat.indices
    return rows, cols


def normalize_rows(a):
    a /= a.sum(1)[:,None]
    return a


def log_likelihood(data, wordprobs):
    return np.sum(np.nan_to_num(np.log(wordprobs)) * data.data) \
        + gammaln(data.sum(1)+1).sum() - gammaln(data.data+1).sum()


def check_timestamps(timestamps):
    assert np.all(timestamps == timestamps[np.argsort(timestamps)])


def timeindices_from_timestamps(timestamps):
    return timestamps - timestamps[0]


###
# LDA Models
###

# LDA and the CTMs all treat beta, z, and likelihoods the same way, so that
# stuff is factored out into _LDABase. Since each model treats theta
# differently, theta stuff is left abstract.

class _LDABase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, T, alpha_beta):
        assert isinstance(data, scipy.sparse.csr.csr_matrix)
        self.D, self.V = data.shape
        self.T = T
        self.alpha_beta = alpha_beta

        self.data = data

        self.pyrngs = initialize_pyrngs()

        self.initialize_beta()
        self.initialize_theta()
        self.z = np.zeros((data.data.shape[0], T), dtype='uint32')
        self.resample_z()

        # precompute
        self._training_gammalns = \
            gammaln(data.sum(1)+1).sum() - gammaln(data.data+1).sum()

    @abc.abstractproperty
    def theta(self):
        pass

    @abc.abstractmethod
    def initialize_theta(self):
        pass

    @abc.abstractmethod
    def resample_theta(self):
        pass

    def initialize_beta(self):
        self.beta = sample_dirichlet(
            self.alpha_beta * np.ones((self.V, self.T)), 'vert')

    @abc.abstractproperty
    def copy_sample(self):
        pass

    def get_wordprobs(self, data):
        return self.theta.dot(self.beta.T)[csr_nonzero(data)]

    def get_topicprobs(self, data):
        rows, cols = csr_nonzero(data)
        return normalize_rows(self.theta[rows] * self.beta[cols])

    def log_likelihood(self):
        # this method is separated to avoid recomputing the training gammalns
        wordprobs = self.get_wordprobs(self.data)
        return np.sum(np.nan_to_num(np.log(wordprobs)) * self.data.data) \
            + self._training_gammalns

    def heldout_log_likelihood(self, data):
        return log_likelihood(data, self.get_wordprobs(data))

    def perplexity(self, data):
        return np.exp(-self.heldout_log_likelihood(data)
                      / data.sum())

    def resample(self):
        self.resample_z()
        self.resample_theta()
        self.resample_beta()

    def resample_beta(self):
        self.beta = sample_dirichlet(
            self.alpha_beta + self.word_topic_counts, 'v')

    def resample_z(self):
        topicprobs = self.get_topicprobs(self.data)
        multinomial_par(self.pyrngs, self.data.data, topicprobs, self.z)
        self._update_counts()

    def _update_counts(self):
        self.doc_topic_counts = np.zeros((self.D, self.T), dtype='uint32')
        self.word_topic_counts = np.zeros((self.V, self.T), dtype='uint32')
        rows, cols = csr_nonzero(self.data)
        for i, j, zvec in zip(rows, cols, self.z):
            self.doc_topic_counts[i] += zvec
            self.word_topic_counts[j] += zvec


class StandardLDA(_LDABase):
    "Standard LDA with Dirichlet priors"

    def __init__(self, data, T, alpha_beta, alpha_theta):
        self.alpha_theta = alpha_theta
        super(StandardLDA, self).__init__(data, T, alpha_beta)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    def initialize_theta(self):
        self.theta = sample_dirichlet(
            self.alpha_theta * np.ones((self.D, self.T)), 'horiz')

    def resample_theta(self):
        self.theta = sample_dirichlet(
            self.alpha_theta + self.doc_topic_counts, 'horiz')

    def copy_sample(self):
        new = copy.copy(self)
        new.beta = self.beta.copy()
        new._theta = self._theta.copy()
        return new

    def resample_collapsed(self,niter=1):
        self.resample_z_collapsed(niter)
        self.resample_theta()
        self.resample_beta()

    def resample_z_collapsed(self,niter=1):
        from _lda import CollapsedCounts

        counts = CollapsedCounts(
            self.alpha_theta, self.alpha_beta, self.T,
            self.z, self.doc_topic_counts, self.word_topic_counts,
            self.data)
        counts.resample(niter)

        self.z = counts.z
        self.doc_topic_counts = counts.doc_topic_counts
        self.word_topic_counts = counts.word_topic_counts


###
# Correlated LDA Models (CTMs)
###

class StickbreakingCorrelatedLDA(_LDABase):
    "Correlated LDA with the stick breaking representation"

    def __init__(self, data, T, alpha_beta):
        mu, sigma = compute_uniform_mean_psi(T)
        self.theta_prior = Gaussian(
            mu=mu, sigma=sigma, mu_0=mu, sigma_0=T*sigma/10.,
            nu_0=T/10., kappa_0=1./10)

        self.ppgs = initialize_polya_gamma_samplers()
        self.omega = np.zeros((data.shape[0], T-1))

        super(StickbreakingCorrelatedLDA, self).__init__(data, T, alpha_beta)

    @property
    def theta(self):
        return psi_to_pi(self.psi)

    @theta.setter
    def theta(self, theta):
        self.psi = pi_to_psi(theta)

    def initialize_theta(self):
        self.psi = np.tile(self.theta_prior.mu, (self.D, 1))

    def resample_theta(self):
        self.resample_omega()
        self.resample_psi()

    def resample(self):
        super(StickbreakingCorrelatedLDA, self).resample()
        self.resample_theta_prior()

    def resample_omega(self):
        pgdrawvpar(
            self.ppgs, N_vec(self.doc_topic_counts).astype('float64').ravel(),
            self.psi.ravel(), self.omega.ravel())
        np.clip(self.omega, 1e-32, np.inf, out=self.omega)

    def resample_psi(self):
        mu = self.theta_prior.mu
        Lmbda = np.linalg.inv(self.theta_prior.sigma)
        randvec = np.random.randn(self.D, self.T-1)

        for d, c in enumerate(self.doc_topic_counts):
            Lmbda_post = Lmbda + np.diag(self.omega[d])
            h_post = Lmbda.dot(mu) + kappa_vec(c)
            L = np.linalg.cholesky(Lmbda_post)
            self.psi[d] = dpotrs(L, h_post) + solve_triangular(L, randvec[d])

    def resample_theta_prior(self):
        self.theta_prior.resample(self.psi)

    def copy_sample(self):
        new = copy.copy(self)
        new.beta = self.beta.copy()
        new.psi = self.psi.copy()
        new.theta_prior = self.theta_prior.copy_sample()
        del new.z
        del new.omega
        return new


class LogisticNormalCorrelatedLDA(_LDABase):
    "Correlated LDA with the stick breaking representation"

    def __init__(self, data, T, alpha_beta):
        mu, sigma = np.zeros(T), np.eye(T)
        self.theta_prior = \
            Gaussian(
                mu=mu, sigma=sigma, mu_0=mu, sigma_0=T*sigma/10.,
                nu_0=T/10., kappa_0=10.)

        self.ppgs = initialize_polya_gamma_samplers()
        self.omega = np.zeros((data.shape[0], T))

        super(LogisticNormalCorrelatedLDA, self).__init__(data, T, alpha_beta)

    @property
    def theta(self):
        return ln_psi_to_pi(self.psi)

    @theta.setter
    def theta(self, theta):
        self.psi = ln_pi_to_psi(theta)

    def initialize_theta(self):
        self.psi = np.tile(self.theta_prior.mu, (self.D, 1))

    def resample_theta(self):
        self.resample_psi_and_omega()

    def resample(self):
        super(LogisticNormalCorrelatedLDA, self).resample()
        self.resample_theta_prior()

    def resample_psi_and_omega(self):
        Lmbda = np.linalg.inv(self.theta_prior.sigma)
        for d in xrange(self.D):
            N = self.data[d].sum()
            c = self.doc_topic_counts[d]
            for t in xrange(self.T):
                self.omega[d,t] = self.ppgs[0].pgdraw(
                    N, self._conditional_omega(d,t))

                mu_cond, sigma_cond = self._conditional_psi(d, t, Lmbda, N, c)
                self.psi[d,t] = np.random.normal(mu_cond, np.sqrt(sigma_cond))

    def _conditional_psi(self, d, t, Lmbda, N, c):
        nott = np.arange(self.T) != t
        psi = self.psi[d]
        omega = self.omega[d]
        mu = self.theta_prior.mu

        zetat = logsumexp(psi[nott])

        mut_marg = mu[t] - 1./Lmbda[t,t] * Lmbda[t,nott].dot(psi[nott] - mu[nott])
        sigmat_marg = 1./Lmbda[t,t]

        sigmat_cond = 1./(omega[t] + 1./sigmat_marg)

        # kappa is the mean dot precision, i.e. the sufficient statistic of a Gaussian
        # therefore we can sum over datapoints
        kappa = (c[t] - N/2.0).sum()
        mut_cond = sigmat_cond * (kappa + mut_marg / sigmat_marg + omega[t]*zetat)

        return mut_cond, sigmat_cond

    def _conditional_omega(self, d, t):
        nott = np.arange(self.T) != t
        psi = self.psi[d]
        zetat = logsumexp(psi[nott])
        return psi[t] - zetat

    def resample_theta_prior(self):
        self.theta_prior.resample(self.psi)

    def copy_sample(self):
        new = copy.copy(self)
        new.beta = self.beta.copy()
        new.psi = self.psi.copy()
        new.theta_prior = self.theta_prior.copy_sample()
        del new.z
        del new.omega
        return new


###
# Dynamic LDA Models (DTMs)
###

class StickbreakingDynamicProportionsLDA(_LDABase):
    def __init__(self, data, timestamps, T, alpha_beta):
        check_timestamps(timestamps)
        self.timestamps = timestamps
        self.timeindices = timeindices_from_timestamps(timestamps)

        # TODO initialize LDS parameters (A,B,C,D)

        self.ppgs = initialize_polya_gamma_samplers()
        self.omega = np.zeros((data.shape[0], T-1))

        super(StickbreakingDynamicProportionsLDA, self).__init__(data, T, alpha_beta)

    @property
    def theta(self):
        return psi_to_pi(self.psi)

    @theta.setter
    def theta(self, theta):
        self.psi = pi_to_psi(theta)

    def initialize_theta(self):
        raise NotImplementedError('initialize psi')  # TODO

    def resample(self):
        super(StickbreakingDynamicProportionsLDA, self).resample()
        self.resample_lds_params()

    def resample_omega(self):
        pgdrawvpar(
            self.ppgs, N_vec(self.doc_topic_counts).astype('float64').ravel(),
            self.psi.ravel(), self.omega.ravel())
        np.clip(self.omega, 1e-32, np.inf, out=self.omega)

    def resample_psi(self):
        # only trick here is to compute the potentials, which will include some
        # missing observations and some repeated observations.
        # that is, for each document we have a potential, and we just have to
        # add them up in the right place
        raise NotImplementedError('resample LDS states')  # TODO

    def resample_lds_params(self):
        raise NotImplementedError('resample LDS parameters')  # TODO

    def copy_sample(self):
        raise NotImplementedError


# In the following class, there is a distinct beta matrix for each time index,
# so handling of beta (including computing wordprobs and topicprobs) is handled
# slightly differently. Also, methods for initializing and updating beta are
# overridden, unlike in all the other classes here.

class StickbreakingDynamicTopicsLDA(object):
    pass
