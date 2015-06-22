"""
Some simple models with multinomial observations and Gaussian priors.
"""
import copy
import numpy as np
from numpy.lib.stride_tricks import as_strided

from scipy.special import gammaln
from scipy.misc import logsumexp

from pybasicbayes.abstractions import Model, ModelGibbsSampling
from pylds.states import LDSStates
from pylds.lds_messages_interface import \
    filter_and_sample, filter_and_sample_diagonal, \
    kalman_filter, kalman_info_filter, kalman_filter_diagonal

from pgmult.distributions import PGMultinomialRegression
from pgmult.internals.utils import N_vec, kappa_vec

class MultinomialLDSStates(LDSStates):

    def old_resample(self, conditional_mean, conditional_cov):
        # Filter and sample with with heterogeneous noise
        # from the Polya gamma observations
        assert isinstance(self.model.emission_distn, PGMultinomialRegression)
        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_cov.shape == (self.T, self.p, self.p)

        ll, self.stateseq = filter_and_sample(
            self.mu_init, self.sigma_init,
            self.A,
            self.sigma_states,
            self.C,
            conditional_cov,
            conditional_mean)

        # print "states:  (", self.stateseq.min(), ", ", self.stateseq.max(), ")"

        assert np.all(np.isfinite(self.stateseq))

    def resample(self, conditional_mean, conditional_cov):
        # Filter and sample with with heterogeneous noise
        # from the Polya gamma observations
        assert isinstance(self.model.emission_distn, PGMultinomialRegression)
        C = self.model.emission_distn.C

        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_cov.shape == (self.T, self.p)

        ll, self.stateseq = filter_and_sample_diagonal(
            self.mu_init, self.sigma_init,
            self.A,
            self.sigma_states,
            self.C,
            conditional_cov,
            conditional_mean)

        # print "states:  (", self.stateseq.min(), ", ", self.stateseq.max(), ")"

        assert np.all(np.isfinite(self.stateseq))


    def log_likelihood(self, conditional_mean, conditional_cov):
        assert isinstance(self.model.emission_distn, PGMultinomialRegression)
        C = self.model.emission_distn.C

        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_cov.shape == (self.T, self.p, self.p)

        # Use stride tricks to replicate A, sigma_states, and C
        # np.tile(self.A[None, :, :], (self.T, 1, 1))
        A_strided = as_strided(self.A, shape=(self.T, self.n, self.n),
                               strides=(0,) + self.A.strides)

        # np.tile(self.sigma_states[None,:,:], (self.T, 1, 1))
        sigma_states_strided = \
            as_strided(self.sigma_states, shape=(self.T, self.n, self.n),
                       strides=(0,) + self.sigma_states.strides)

        # np.tile(C[None,:,:], (self.T,1,1))
        C_strided = as_strided(C, shape=(self.T, self.p, self.n),
                               strides=(0,) + C.strides)


        normalizer, _, _ = kalman_filter(
            self.mu_init, self.sigma_init,
            A_strided,
            sigma_states_strided,
            C_strided,
            conditional_cov,
            conditional_mean)
        return normalizer

    def info_log_likelihood(self, conditional_mean, conditional_prec):
        assert isinstance(self.model.emission_distn, PGMultinomialRegression)
        C = self.model.emission_distn.C

        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_prec.shape == (self.T, self.p, self.p)

        # Compute the info form potentials
        # Initial distribution potentials
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        # Transition potentials
        J_22 = np.linalg.inv(self.sigma_states)
        J_21 = -(J_22.dot(self.A))
        J_11 = self.A.T.dot(J_22.dot(self.A))

        # Use stride tricks to replicate A, sigma_states, and C
        J_11_strided = as_strided(J_11, shape=(self.T, self.n, self.n),
                                  strides=(0,) + J_11.strides)

        J_21_strided = as_strided(J_21, shape=(self.T, self.n, self.n),
                                  strides=(0,) + J_21.strides)

        J_22_strided = as_strided(J_22, shape=(self.T, self.n, self.n),
                                  strides=(0,) + J_22.strides)

        # Observation potentials
        J_node = np.array([C.T.dot(O).dot(C) for O in conditional_prec])
        h_node = np.array([mu.dot(O).dot(C) for mu in conditional_mean])

        # Compute the (unnormalized) log likelihood
        ll, filtered_Js, filtered_hs = kalman_info_filter(
            J_init, h_init,
            J_11_strided, J_21_strided, J_22_strided,
            J_node, h_node)

        # Compute the extra normalizer terms
        p, n = C.shape
        T = conditional_mean.shape[0]

        ll -= 1./2 * self.mu_init.dot(h_init)
        ll -= 1./2 * np.linalg.slogdet(self.sigma_init)[1]
        ll -= n/2. * np.log(2*np.pi)

        ll -= (T-1)/2. * np.linalg.slogdet(self.A)[1]
        ll -= (T-1)*n/2. * np.log(2*np.pi)

        # # TODO: Decide whether or not to compute terms for likelihood
        # for t in xrange(T):
        #     O = conditional_prec[t]
        #     ll -= 1./2 * np.einsum('ij,i,j->',O,conditional_mean[t],conditional_mean[t])
        #     O = np.diag(np.clip(np.diag(O), 1e-16,np.inf))
        #     ll -= 1/2. * -(np.linalg.slogdet(O)[1])
        #     ll -= 1*p/2 * np.log(2*np.pi)

        return ll

    def predict_states(self, conditional_mean, conditional_cov, Tpred=1, Npred=1):
        assert isinstance(self.model.emission_distn, PGMultinomialRegression)
        assert conditional_mean.shape == (self.T, self.p)
        assert conditional_cov.shape == (self.T, self.p)

        A = self.A
        sigma_states = self.sigma_states

        _, filtered_mus, filtered_sigmas = kalman_filter_diagonal(
            self.mu_init, self.sigma_init, self.A, self.sigma_states, self.C,
            conditional_cov, conditional_mean)
        init_mu = A.dot(filtered_mus[-1])
        init_sigma = sigma_states + A.dot(filtered_sigmas[-1]).dot(A.T)

        # Sample Npred arrays of size Tpred
        # (using random Gaussians for the last Tpred -1)
        randseq = np.einsum('tjN,ij->tiN',
                            np.random.randn(Tpred-1, self.n, Npred),
                            np.linalg.cholesky(sigma_states))

        out = np.zeros((Tpred, self.n, Npred))
        out[0,:,:] = np.random.multivariate_normal(init_mu, init_sigma, size=Npred).T
        for t in range(1,Tpred):
            out[t] = A.dot(out[t-1]) + randseq[t-1]

        if Tpred > 1:
            assert np.allclose(out[1,:,0], A.dot(out[0,:,0]) + randseq[0,:,0])

        return out

class _MultinomialLDSBase(Model):
    def __init__(self, K, n,
                 dynamics_distn,
                 init_dynamics_distn,
                 C=None, sigma_C=1.0, mu_pi=None):
        self.K = K
        self.n = n

        self.init_dynamics_distn = init_dynamics_distn
        self.mu_init = self.init_dynamics_distn.mu
        self.sigma_init = self.init_dynamics_distn.sigma
        self.dynamics_distn = dynamics_distn

        if mu_pi is None:
            mu_pi = np.ones(self.K)/self.K

        self.emission_distn = PGMultinomialRegression(K, n, C=C, sigma_C=sigma_C,
                                                      mu_pi=mu_pi)

        # Initialize a list of augmented data dicts
        self.data_list = []

    @property
    def p(self):
        return self.K-1

    @property
    def A(self):
        return self.dynamics_distn.A

    @A.setter
    def A(self,A):
        self.dynamics_distn.A = A

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @sigma_states.setter
    def sigma_states(self,sigma_states):
        self.dynamics_distn.sigma = sigma_states

    @property
    def C(self):
        return self.emission_distn.C

    @C.setter
    def C(self,C):
        self.emission_distn.C = C

    @property
    def mu_init(self):
        return self.init_dynamics_distn.mu

    @mu_init.setter
    def mu_init(self, mu):
        self.init_dynamics_distn.mu = mu

    @property
    def sigma_init(self):
        return self.init_dynamics_distn.sigma

    @sigma_init.setter
    def sigma_init(self, sigma):
        self.init_dynamics_distn.sigma = sigma

    def add_data(self, data):
        assert isinstance(data, np.ndarray) \
               and data.ndim == 2 \
               and data.shape[1] == self.K
        T = data.shape[0]

        augmented_data = {"x": data, "T": T}

        # Create a states object
        augmented_data["states"] = MultinomialLDSStates(model=self, data=data)

        self.emission_distn.augment_data(augmented_data)

        self.data_list.append(augmented_data)

    def log_likelihood(self):
        ll = 0
        for data in self.data_list:
            ll += self.emission_distn.log_likelihood(data)
        return ll

    def heldout_log_likelihood(self, X, M=100):
        return self._mc_heldout_log_likelihood(X, M)

    def _info_form_heldout_log_likelihood(self, X, M=10):
        """
        We can analytically integrate out z (latent states)
        given omega. To estimate the heldout log likelihood of a
        data sequence, we Monte Carlo integrate over omega,
        where omega is drawn from the prior.
        :param data:
        :param M: number of Monte Carlo samples for integrating out omega
        :return:
        """
        # assert len(self.data_list) == 1, "TODO: Support more than 1 data set"

        T, K = X.shape
        assert K == self.K
        kappa = kappa_vec(X)
        N = N_vec(X)

        # Compute the data-specific normalization constant from the
        # augmented multinomial distribution
        Z_mul = (gammaln(N + 1) - gammaln(X[:,:-1]+1) - gammaln(N-X[:,:-1]+1)).sum()
        Z_mul += (-N * np.log(2.)).sum()

        # Monte carlo integrate wrt omega ~ PG(N, 0)
        import pypolyagamma as ppg
        hlls = np.zeros(M)
        for m in xrange(M):
            # Sample omega using the emission distributions samplers
            omega = np.zeros(N.size)
            ppg.pgdrawvpar(self.emission_distn.ppgs,
                           N.ravel(), np.zeros(N.size),
                           omega)
            omega = omega.reshape((T, K-1))

            # TODO: Remove this
            #  # Compute the normalization constant for this omega
            # Z_omg = 0.5 * (kappa**2/omega).sum()
            # Z_omg += T * (K-1)/2. * np.log(2*np.pi)
            # Z_omg += -0.5 * np.log(omega).sum()         # 1/2 log det of Omega_t^{-1}

            # Exactly integrate out the latent states z using message passing
            # The "data" is the normal potential from the
            states = MultinomialLDSStates(model=self, data=X)
            conditional_mean = kappa / np.clip(omega, 1e-64,np.inf) - self.emission_distn.mu[None, :]
            conditional_prec = np.zeros((T, K-1, K-1))
            # conditional_cov = np.zeros((T, K-1, K-1))
            for t in xrange(T):
                conditional_prec[t,:,:] = np.diag(omega[t,:])
                # conditional_cov[t,:,:] = np.diag(1./omega[t,:])

            Z_lds = states.info_log_likelihood(conditional_mean, conditional_prec)
            # Z_lds_distn = states.log_likelihood(conditional_mean, conditional_cov)

            # print ""
            # print "Z_mul: ", Z_mul
            # print "Info LDS: ", Z_lds
            # print "Dist LDS: ", Z_lds_distn
            # print "Z_omg: ", Z_omg
            # print "hll: ", (Z_mul + Z_lds + Z_omg)

            # Sum them up to get the heldout log likelihood for this omega
            # hlls[m] = Z_mul + Z_lds + Z_omg
            hlls[m] = Z_mul + Z_lds

        # Now take the log of the average to get the log likelihood
        hll = logsumexp(hlls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(hlls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll

    def _distn_form_heldout_log_likelihood(self, X, M=10):
        """
        We can analytically integrate out z (latent states)
        given omega. To estimate the heldout log likelihood of a
        data sequence, we Monte Carlo integrate over omega,
        where omega is drawn from the prior.
        :param data:
        :param M: number of Monte Carlo samples for integrating out omega
        :return:
        """
        # assert len(self.data_list) == 1, "TODO: Support more than 1 data set"

        T, K = X.shape
        assert K == self.K
        kappa = kappa_vec(X)
        N = N_vec(X)

        # Compute the data-specific normalization constant from the
        # augmented multinomial distribution
        Z_mul = (gammaln(N + 1) - gammaln(X[:,:-1]+1) - gammaln(N-X[:,:-1]+1)).sum()
        Z_mul += (-N * np.log(2.)).sum()

        # Monte carlo integrate wrt omega ~ PG(N, 0)
        import pypolyagamma as ppg
        hlls = np.zeros(M)
        for m in xrange(M):
            # Sample omega using the emission distributions samplers
            omega = np.zeros(N.size)
            ppg.pgdrawvpar(self.emission_distn.ppgs,
                           N.ravel(), np.zeros(N.size),
                           omega)
            omega = omega.reshape((T, K-1))
            # valid = omega > 0
            omega = np.clip(omega, 1e-8, np.inf)

            # Compute the normalization constant for this omega
            Z_omg = 0.5 * (kappa**2/omega).sum()
            # Z_omg = 0.5 * (kappa[valid]**2/omega[valid]).sum()
            Z_omg += T * (K-1)/2. * np.log(2*np.pi)
            # Z_omg += valid.sum()/2. * np.log(2*np.pi)
            Z_omg += -0.5 * np.log(omega).sum()         # 1/2 log det of Omega_t^{-1}
            # Z_omg += -0.5 * np.log(omega[valid]).sum()         # 1/2 log det of Omega_t^{-1}

            # clip omega = zero for message passing
            # omega[~valid] = 1e-32

            # Exactly integrate out the latent states z using message passing
            # The "data" is the normal potential from the
            states = MultinomialLDSStates(model=self, data=X)
            conditional_mean = kappa / omega - self.emission_distn.mu[None, :]
            # conditional_mean[~np.isfinite(conditional_mean)] = 0
            conditional_cov = np.zeros((T, K-1, K-1))
            for t in xrange(T):
                conditional_cov[t,:,:] = np.diag(1./omega[t,:])
            Z_lds = states.log_likelihood(conditional_mean, conditional_cov)

            # Sum them up to get the heldout log likelihood for this omega
            hlls[m] = Z_mul + Z_omg + Z_lds

            # print ""
            # print "Z_mul: ", Z_mul
            # print "DistLDS: ", Z_lds
            # print "Z_omg: ", Z_omg
            # print "hll: ", (Z_mul + Z_lds + Z_omg)

        # Now take the log of the average to get the log likelihood
        hll = logsumexp(hlls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(hlls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll


    def _mc_heldout_log_likelihood(self, X, M=100):
        # Estimate the held out likelihood using Monte Carlo
        T, K = X.shape
        assert K == self.K

        lls = np.zeros(M)
        for m in xrange(M):
            # if m % 100 == 0:
            #     print "Iteration ", m

            # Sample latent states from the prior
            data = self.generate(T=T, keep=False)
            data["x"] = X
            lls[m] = self.emission_distn.log_likelihood(data)

        # Compute the average
        hll = logsumexp(lls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll

    def predictive_log_likelihood(self, X_pred, data_index=0, M=100):
        """
        Hacky way of computing the predictive log likelihood
        :param X_pred:
        :param data_index:
        :param M:
        :return:
        """
        Tpred = X_pred.shape[0]

        data = self.data_list[data_index]
        conditional_mean = self.emission_distn.conditional_mean(data)
        conditional_cov = self.emission_distn.conditional_cov(data, flat=True)

        lls = []
        z_preds = data["states"].predict_states(
                conditional_mean, conditional_cov, Tpred=Tpred, Npred=M)
        for m in xrange(M):
            ll_pred = self.emission_distn.log_likelihood(
                {"z": z_preds[...,m], "x": X_pred})
            lls.append(ll_pred)

        # Compute the average
        hll = logsumexp(lls) - np.log(M)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, M), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(M)
        std_hll = hll_samples.std()

        return hll, std_hll

    def pi(self, data):
        return self.emission_distn.pi(data)

    def psi(self, data):
        return self.emission_distn.psi(data)


    def generate(self, T=100, N=1, keep=True):
        augmented_data = {"T": T}

        # Sample latent state sequence
        states = MultinomialLDSStates(model=self, T=T, initialize_from_prior=True)
        augmented_data["states"] = states

        # Sample observations
        x = self.emission_distn.rvs(z=states.stateseq, N=N)
        states.data = x
        augmented_data["x"] = x

        if keep:
            self.data_list.append(augmented_data)

        return augmented_data

    def rvs(self,T):
        return self.generate(keep=False, T=T)[0]


class _MultinomialLDSGibbsSampling(_MultinomialLDSBase, ModelGibbsSampling):
    def resample_model(self):
        self.resample_parameters()
        self.resample_states()

    def resample_states(self):
        for data in self.data_list:
            conditional_mean = self.emission_distn.conditional_mean(data)
            conditional_cov = self.emission_distn.conditional_cov(data, flat=True)
            data["states"].resample(conditional_mean, conditional_cov)

        # for data in self.data_list:
        #     conditional_mean = self.emission_distn.conditional_mean(data)
        #     conditional_cov = self.emission_distn.conditional_cov(data, flat=False)
        #     data["states"].old_resample(conditional_mean, conditional_cov)

    def resample_parameters(self):
        self.resample_init_dynamics_distn()
        self.resample_dynamics_distn()
        self.resample_emission_distn()

    def resample_init_dynamics_distn(self):
        self.init_dynamics_distn.resample(
            [np.atleast_2d(d["states"].stateseq[0]) for d in self.data_list])

    def resample_dynamics_distn(self):
        self.dynamics_distn.resample(
            [d["states"].strided_stateseq for d in self.data_list])

    def resample_emission_distn(self):
        self.emission_distn.resample(self.data_list)

    def initialize_from_gaussian_lds(self, N_samples=100):
        """
        Initialize z, A, C, sigma_states using a Gaussian LDS
        :return:
        """
        from pylds.models import DefaultLDS
        init_model = DefaultLDS(n=self.n, p=self.K)

        for data in self.data_list:
            init_model.add_data(data["x"])

        print "Initializing with Gaussian LDS"
        # for smpl in xrange(N_samples):
        #     init_model.resample_model()
        for smpl in xrange(20):
            init_model.resample_model()

        # Use the init model's parameters
        self.A = init_model.A.copy()
        self.C = init_model.C[:self.K-1,:].copy()
        self.sigma_states = init_model.sigma_states.copy()
        self.mu_init = init_model.mu_init.copy()
        self.sigma_init = init_model.sigma_init.copy()

        # Use the init model's latent state sequences too
        for data, init_data in zip(self.data_list, init_model.states_list):
            data["z"] = init_data.stateseq.copy()

        # Now resample omega
        self.emission_distn.resample_omega(self.data_list)

    def initialize_from_data(self):
        """
        Initialize the observation mean to the mean of the data
        :return:
        """
        raise NotImplementedError


class MultinomialLDS(_MultinomialLDSGibbsSampling):
    pass

