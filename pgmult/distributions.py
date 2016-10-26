
import numpy as np
from numpy.linalg import inv, solve

from scipy.special import gammaln
from scipy.misc import logsumexp

from pybasicbayes.abstractions import Distribution, GibbsSampling, Model
from pybasicbayes.distributions import Multinomial

from pgmult.utils import kappa_vec, N_vec, pi_to_psi, psi_to_pi, \
    ln_psi_to_pi, ln_pi_to_psi, initialize_polya_gamma_samplers, \
    compute_psi_cmoments

import pypolyagamma as ppg

class PGMultinomial(GibbsSampling):
    """
    A base class for the Polya-gamma augmented multinomial distribution.
    The parameter of the multinomial distribution, \pi, is obtained by
    transforming the Gaussian-distributed vector, \psi. To perform inference
    over \psi given multinomial observations, we augment the distribution
    with a Polya-gamma distributed vector \omega. The transformation from
    \psi to \pi is given by:

    \pi_1 = \sigma(\psi_1)
    \pi_k = \sigma(\psi_k) (1-\sum_{j < k} \pi_j)  for k = 2..K-1
    \pi_K = 1-\sum_{j<K} \pi_j

    where \sigma is the logistic function mapping reals to [0,1].
    """

    def __init__(self, K, pi=None, psi=None, mu=None, Sigma=None):
        """
        Create a PGMultinomial distribution with mean and covariance for psi.

        :param K:       Dimensionality of the multinomial distribution
        :param pi:      Multinomial probability vector (must sum to 1)
        :param psi:     Transformed multinomial probability vector
        :param mu:      Mean of \psi
        :param Sigma:   Covariance of \psi
        """
        assert isinstance(K, int) and K >= 2, "K must be an integer >= 2"
        self.K = K

        if all(param is None for param in (pi,psi,mu,Sigma)):
            mu, sigma = compute_psi_cmoments(np.ones(K))
            Sigma = np.diag(sigma)

        if pi is not None:
            if not (isinstance(pi, np.ndarray) and  pi.shape == (K,)
                    and np.isclose(pi.sum(), 1.0)):
                raise ValueError("Pi must be a normalized length-K vector")
            self.pi = pi

        if psi is not None:
            if not (isinstance(psi, np.ndarray) and psi.shape == (K-1,)):
                raise ValueError("Psi must be a (K-1) vector of reals")
            self.psi = psi

        if mu is not None and Sigma is not None:
            if not (isinstance(mu, np.ndarray) and mu.shape == (K-1,)):
                raise ValueError("Mu must be a (K-1) vector")
            if not (isinstance(Sigma, np.ndarray) and Sigma.shape == ((K-1), (K-1))):
                raise ValueError("Sigma must be a K-1 Covariance matrix")
            self.mu = mu
            self.Sigma = Sigma

            # If psi and pi have not been given, sample from the prior
            if psi is None and pi is None:
                self.psi = np.random.multivariate_normal(self.mu, self.Sigma)

        # Initialize Polya-gamma augmentation variables
        self.ppgs = initialize_polya_gamma_samplers()
        self.omega = np.ones(self.K-1)

    @property
    def pi(self):
        return psi_to_pi(self.psi)

    @pi.setter
    def pi(self, value):
        self.psi = pi_to_psi(value)

    def log_likelihood(self, x):
        ll = 0
        ll += gammaln((x+1).sum()) - gammaln(x+1).sum()
        ll += (x * np.log(self.pi)).sum()
        return ll

    def rvs(self, size=1, N=1):
        """
        Sample from a PG augmented multinomial distribution
        :param size:
        :return:
        """
        # assert self.mu is not None and self.Sigma is not None, "mu and sigma are not specified!"
        # psis = np.random.multivariate_normal(self.mu, self.Sigma, size=size)
        # pis = np.empty((size, self.K))
        # for i in xrange(size):
        #     pis[i,:] = psi_to_pi(psis[i,:])
        # return pis

        # Sample from the multinomial distribution
        return np.random.multinomial(N, self.pi, size=size)

    def resample(self, x=None):
        if x is None:
            x = np.zeros((0,self.K))

        self.resample_omega(x)
        self.resample_psi(x)

    def conditional_psi(self, x):
        """
        Compute the conditional distribution over psi given observation x and omega
        :param x:
        :return:
        """
        assert x.ndim == 2
        Omega = np.diag(self.omega)
        Sigma_cond = inv(Omega + inv(self.Sigma))

        # kappa is the mean dot precision, i.e. the sufficient statistic of a Gaussian
        # therefore we can sum over datapoints
        kappa = kappa_vec(x).sum(0)
        mu_cond = Sigma_cond.dot(kappa +
                                 solve(self.Sigma, self.mu))

        return mu_cond, Sigma_cond

    def resample_psi(self, x):
        mu_cond, Sigma_cond = self.conditional_psi(x)
        self.psi = np.random.multivariate_normal(mu_cond, Sigma_cond)

    def resample_omega(self, x):
        """
        Resample omega from its conditional Polya-gamma distribution
        :return:
        """
        assert x.ndim == 2
        N = N_vec(x)

        #  Sum the N's (i.e. the b's in the denominator)
        NN = N.sum(0).astype(np.float)
        ppg.pgdrawvpar(self.ppgs, NN, self.psi, self.omega)


class PGMultinomialRegression(Distribution):
    """
    z    ~ Norm(.,.)     eg. Latent state of an LDS
    x    ~ Mult(N, Cz)
    """
    def __init__(self, K, n, C=None, sigma_C=1, mu=None, mu_pi=None):
        """
        Create a PGMultinomial distribution with mean and covariance for psi.

        :param K:       Dimensionality of the multinomial distribution
        :param mu_C:    Mean of the matrix normal distribution over C
        """
        assert isinstance(K, int) and K >= 2, "K must be an integer >= 2"
        self.K = K

        assert isinstance(n, int) and n >= 1, "n must be an integer >= 1"
        self.n = n

        # Initialize emission matrix C
        self.sigma_C = sigma_C
        if C is None:
            self.C = self.sigma_C * np.random.randn(self.K-1, self.n)
            # mu, sigma = compute_psi_cmoments(np.ones(K))
            # self.C = compute_psi_cmoments(np.ones(K))[0][:,None] * np.ones((self.K-1, self.n))
        else:
            assert C.shape == (self.K-1, self.n)
            self.C = C

        # Initialize the observation mean (mu)
        if mu is None and mu_pi is None:
            self.mu = np.zeros(self.K-1)
        elif mu is not None:
            assert mu.shape == (self.K-1,)
            self.mu = mu
        else:
            assert mu_pi.shape == (self.K,)
            self.mu = pi_to_psi(mu_pi)

        # Initialize Polya-gamma augmentation variables
        self.ppgs = initialize_polya_gamma_samplers()

    def augment_data(self, augmented_data):
        """
        Augment the data with auxiliary variables
        :param augmented_data:
        :return:
        """
        x = augmented_data["x"]
        T, K = x.shape
        assert K == self.K

        augmented_data["kappa"] = kappa_vec(x)
        augmented_data["omega"] = np.ones((T,K-1))

        self.resample_omega([augmented_data])

        return augmented_data

    def psi(self, data):
        # TODO: Fix this hack
        if "z" in data:
            z = data["z"]
        elif "states" in data:
            z = data["states"].stateseq
        else:
            raise Exception("Could not find latent states!")

        psi = z.dot(self.C.T) + self.mu[None,:]
        return psi

    def pi(self, data):
        psi = self.psi(data)
        # pi = np.array([psi_to_pi(p) for p in psi])
        pi = psi_to_pi(psi)
        return pi

    def log_likelihood(self, data):
        x = data["x"]
        pi = self.pi(data)
        pi = np.clip(pi, 1e-16, 1-1e-16)

        # Compute the multinomial log likelihood given psi
        assert x.shape == pi.shape
        ll = 0
        ll += gammaln(x.sum(axis=1) + 1).sum() - gammaln(x+1).sum()
        ll += (x * np.log(pi)).sum()
        return ll

    def rvs(self, z, N=1, full_output=False):
        """
        Sample from a PG augmented multinomial distribution
        :param size:
        :return:
        """
        T,D = z.shape
        psis = z.dot(self.C.T) + self.mu[None, :]
        pis = np.zeros((T, self.K))
        xs = np.zeros((T, self.K))
        for t in range(T):
            pis[t,:] = psi_to_pi(psis[t,:])
            xs[t,:] = np.random.multinomial(N, pis[t,:])

        if full_output:
            return pis, xs
        else:
            return xs

    def resample(self, augmented_data_list):
        self.resample_C(augmented_data_list)
        self.resample_omega(augmented_data_list)

    def resample_C(self, augmented_data_list):
        """
        Resample the observation vectors. Since the emission noise is diagonal,
        we can resample the rows of C independently
        :return:
        """
        # Get the prior
        prior_precision = 1./self.sigma_C * np.eye(self.n)
        prior_mean = np.zeros(self.n)
        prior_mean_dot_precision = prior_mean.dot(prior_precision)

        # Get the sufficient statistics from the likelihood
        lkhd_precision = np.zeros((self.K-1, self.n, self.n))
        lkhd_mean_dot_precision = np.zeros((self.K-1, self.n))

        for data in augmented_data_list:
            # Compute the residual activation from other components
            # TODO: Fix this hack
            if "z" in data:
                z = data["z"]
            elif "states" in data:
                z = data["states"].stateseq
            else:
                raise Exception("Could not find latent states in augmented data!")

            # Get the observed mean and variance
            omega = data["omega"]
            kappa = data["kappa"]
            prec_obs = omega
            mu_obs = kappa / omega - self.mu[None, :]
            mu_dot_prec_obs = omega * mu_obs

            # Update the sufficient statistics for each neuron
            for k in range(self.K-1):
                lkhd_precision[k,:,:] += (z * prec_obs[:,k][:,None]).T.dot(z)
                lkhd_mean_dot_precision[k,:] += \
                    (mu_dot_prec_obs[:,k]).T.dot(z)

        # Sample each row of C
        for k in range(self.K-1):
            post_prec = prior_precision + lkhd_precision[k,:,:]
            post_cov  = np.linalg.inv(post_prec)
            post_mu   =  (prior_mean_dot_precision +
                          lkhd_mean_dot_precision[k,:]).dot(post_cov)
            post_mu   = post_mu.ravel()

            self.C[k,:] = np.random.multivariate_normal(post_mu, post_cov)

    def resample_omega(self, augmented_data_list):
        """
        Resample omega from its conditional Polya-gamma distribution
        :return:
        """
        K = self.K
        for data in augmented_data_list:
            x = data["x"]
            T = data["T"]

            # TODO: Fix this hack
            if "z" in data:
                z = data["z"]
            elif "states" in data:
                z = data["states"].stateseq
            else:
                raise Exception("Could not find latent states in augmented data!")

            psi = z.dot(self.C.T) + self.mu[None, :]
            N = N_vec(x).astype(np.float)
            tmp_omg = np.zeros(N.size)
            ppg.pgdrawvpar(self.ppgs, N.ravel(), psi.ravel(), tmp_omg)
            data["omega"] = tmp_omg.reshape((T, self.K-1))

            # Clip out zeros
            data["omega"] = np.clip(data["omega"], 1e-8,np.inf)

    def conditional_mean(self, augmented_data):
        """
        Compute the conditional mean \psi given \omega
        :param augmented_data:
        :return:
        """
        cm = augmented_data["kappa"] / augmented_data["omega"]
        cm[~np.isfinite(cm)] = 0
        cm -= self.mu[None,:]
        return cm

    def conditional_prec(self, augmented_data, flat=False):
        """
        Compute the conditional mean \psi given \omega
        :param augmented_data:
        :return:
        """
        O = augmented_data["omega"]
        T = augmented_data["T"]
        Km1 = self.K-1

        if flat:
            prec = O
        else:
            prec = np.zeros((T, Km1, Km1))
            for t in range(T):
                prec[t,:,:] = np.diag(O[t,:])

        return prec

    def conditional_cov(self, augmented_data, flat=False):
        # Since the precision is diagonal, we can invert elementwise
        O = augmented_data["omega"]
        T = augmented_data["T"]
        Km1 = self.K-1

        if flat:
            cov = 1./O
        else:
            cov = np.zeros((T, Km1, Km1))
            for t in range(T):
                cov[t,:,:] = np.diag(1./O[t,:])

        return cov

### Logistic Normal Models
#   For comparison, we implement the logistic normal model, which is
#   also amenable to PG augmentation, but only the conditional marginals
#   are rendered conjugate with a Gaussian prior, not the conditional joint
#   distribution over \psi_{1:K}.

class PGLogisticNormalMultinomial(GibbsSampling):
    def __init__(self, K, pi=None, psi=None, mu=None, Sigma=None):
        """
        Create a PGMultinomial distribution with mean and covariance for psi.

        :param K:       Dimensionality of the multinomial distribution
        :param pi:      Multinomial probability vector (must sum to 1)
        :param psi:     Transformed multinomial probability vector
        :param mu:      Mean of \psi
        :param Sigma:   Covariance of \psi
        """
        assert isinstance(K, int) and K >= 2, "K must be an integer >= 2"
        self.K = K

        assert pi is not None or psi is not None or None not in (mu, Sigma), \
            "pi, psi, or (mu and Sigma) must be specified"

        if pi is not None:
            assert isinstance(pi, np.ndarray) and \
                   pi.shape == (K,) and \
                   np.allclose(pi.sum(), 1.0), \
                "Pi must be a normalized length-K vector"
            self.pi = pi

        if psi is not None:
            assert isinstance(psi, np.ndarray) and \
                   psi.shape == (K-1,), \
                "Psi must be a length-K vector of reals"
            self.psi = psi

        if None not in (mu, Sigma):
            assert isinstance(mu, np.ndarray) and mu.shape == (K,), \
                "Mu must be a length-K vector"

            assert isinstance(Sigma, np.ndarray) and Sigma.shape == (K,K), \
                "Sigma must be a KxK Covariance matrix"
            self.mu = mu
            self.Sigma = Sigma
            self.Lambda = np.linalg.inv(Sigma)

            # If psi and pi have not been given, sample from the prior
            if psi is None and pi is None:
                self.psi = np.random.multivariate_normal(self.mu, self.Sigma)

        # Initialize Polya-gamma augmentation variables
        self.ppgs   = initialize_polya_gamma_samplers()
        self.omega = np.ones(self.K)

        # Initialize the space for the transformed psi variables, rho
        self.rho = np.zeros(self.K)

    @property
    def pi(self):
        return ln_psi_to_pi(self.psi)

    @pi.setter
    def pi(self, value):
        self.psi = ln_pi_to_psi(value)

    def log_likelihood(self, x):
        ll = 0
        ll += gammaln((x+1).sum()) - gammaln(x+1).sum()
        ll += (x * np.log(self.pi)).sum()
        return ll

    def rvs(self, size=1, N=1):
        """
        Sample from a PG augmented multinomial distribution
        :param size:
        :return:
        """
        # Sample from the multinomial distribution
        return np.random.multinomial(N, self.pi, size=size)

    def resample(self, x=None):
        if x is None:
            x = np.zeros((0,self.K))

        self.resample_omega(x)
        self.resample_psi(x)

    def conditional_psi(self, x, k):
        """
        Compute the conditional distribution over psi given observation x and omega
        Using the notation from
        :param x:
        :return:
        """
        if x.ndim == 1:
            xx = x[None,:]
        else:
            xx = x

        Ck = xx[:,k].sum()
        N = xx.sum()

        notk = np.ones(self.K, dtype=np.bool)
        notk[k] = False

        # Compute zeta = log(\sum_{j \neq k} e^{\psi_j})
        zetak = logsumexp(self.psi[notk])

        # Compute rho
        self.rho[k] = self.psi[k] - zetak

        # Get the marginal distribution over rho under the prior
        muk_marg = self.mu[k] \
                   - 1./self.Lambda[k,k] * self.Lambda[k,notk].\
            dot(self.psi[notk] - self.mu[notk])
        sigmak_marg = 1./self.Lambda[k,k]

        # Compute the conditional posterior given psi[notk] and omega
        omegak = self.omega[k]
        sigmak_cond = 1./(omegak + 1./sigmak_marg)

        # kappa is the mean dot precision, i.e. the sufficient statistic of a Gaussian
        # therefore we can sum over datapoints
        kappa = (Ck - N/2.0).sum()
        muk_cond = sigmak_cond * (kappa + muk_marg / sigmak_marg + omegak*zetak)

        return muk_cond, sigmak_cond

    def resample_psi(self, x):
        for k in range(self.K):
            mu_cond, sigma_cond = self.conditional_psi(x, k)
            self.psi[k] = np.random.normal(mu_cond, np.sqrt(sigma_cond))

    def resample_omega(self, x):
        """
        Resample omega from its conditional Polya-gamma distribution
        :return:
        """
        assert x.ndim == 2
        N = x.sum()

        #  Sum the N's (i.e. the b's in the denominator)
        ppg.pgdrawvpar(self.ppgs, N * np.ones(self.K), self.rho, self.omega)




### Competing models
class IndependentMultinomialsModel(Model):
    """
    Naive model where we assume the data is drawn from a
    set of static multinomial distributions. For example,
    we observe a matrix of NxK counts and assume that each
    row is a sample from a multinomial distribution with
    parameter \pi_n. To estimate \pi_n, we use the empirical
    probability under the training data.
    """
    def __init__(self, X):
        assert X.ndim == 2 and (X >= 0).all()

        self.N, self.K = X.shape
        # Compute the empirical name probabilities for each row
        alpha = 1
        self.pi = (X+alpha).astype(np.float) / (X+alpha).sum(axis=1)[:,None]

        self.multinomials = []
        for pi_n in self.pi:
            self.multinomials.append(Multinomial(weights=pi_n, K=self.K))

    def add_data(self,data):
        raise NotImplementedError

    def generate(self,keep=True,**kwargs):
        raise NotImplementedError

    def predictive_log_likelihood(self, X_test):
        assert X_test.shape == (self.N, self.K)
        ll = 0
        ll += gammaln(X_test.sum(axis=1)+1).sum() - gammaln(X_test+1).sum()
        ll += np.nansum(X_test * np.log(self.pi))
        return ll
