###
#   Multinomial LDS models with particle filtering
###
import numpy as np
from scipy.misc import logsumexp

from pylds.models import NonstationaryLDS
from pylds.states import LDSStates

from hips.inference.mh import mh
from hips.inference.particle_mcmc import InitialDistribution, Proposal, Likelihood
from hips.inference.particle_mcmc import ParticleGibbsAncestorSampling

from pgmult.utils import ln_psi_to_pi, psi_to_pi


class LogisticNormalMultinomialLDSStates(LDSStates):

    def __init__(self, model,T=None,data=None,stateseq=None,
            initialize_from_prior=True):

        super(LogisticNormalMultinomialLDSStates, self).\
            __init__(model, T=T, data=data.copy(), stateseq=stateseq,
                     initialize_from_prior=initialize_from_prior)

    @property
    def pi(self):
        psi = self.stateseq.dot(self.C.T)
        return ln_psi_to_pi(psi)

    def resample(self, N_particles=100, N_iter=1):



        init = _GaussianInitialDistribution(self.mu_init, self.sigma_init)
        prop = _VectorizedLinearGaussianDynamicalSystemProposal(self.A, self.sigma_states)
        lkhd = _CategoricalLikelihood(self.C)

        T, D, z, x = self.T, self.n, self.stateseq, self.data
        pf = ParticleGibbsAncestorSampling(T, N_particles, D)
        pf.initialize(init, prop, lkhd, x, z)

        for i in range(N_iter):
            # Reinitialize with the previous particle
            pf.initialize(init, prop, lkhd, x, z)
            z = np.asarray(pf.sample())

        self.stateseq = z

    def sample_predictions(self, Tpred, Npred, obs_noise=True):
        A, sigma_states, z = self.A, self.sigma_states, self.stateseq
        randseq = np.einsum(
            'tjn,ij->tin',
            np.random.randn(Tpred-1, self.n, Npred),
            np.linalg.cholesky(self.sigma_states))

        states = np.empty((Tpred, self.n, Npred))
        states[0] = np.random.multivariate_normal(A.dot(z[-1]), sigma_states, size=Npred).T
        for t in range(1, Tpred):
            states[t] = self.A.dot(states[t-1]) + randseq[t-1]

        return states


class _GaussianInitialDistribution(InitialDistribution):

    def __init__(self, mu, sigma):
        # Check sizes
        if np.isscalar(mu) and np.isscalar(sigma):
            self.D = 1
            mu = np.atleast_2d(mu)
            sigma = np.atleast_2d(sigma)

        elif mu.ndim == 1 and sigma.ndim == 2:
            assert mu.shape[0] == sigma.shape[0] == sigma.shape[1]
            self.D = mu.shape[0]
            mu = mu.reshape((1,self.D))

        elif mu.ndim == 2 and sigma.ndim == 2:
            assert mu.shape[1] == sigma.shape[0] == sigma.shape[1] and mu.shape[0] == 1
            self.D = mu.shape[1]
        else:
            raise Exception('Invalid shape for mu and sigma')

        self.mu = mu
        self.sigma = sigma
        self.chol = np.linalg.cholesky(self.sigma)

    def sample(self, N=1):
        smpls = np.tile(self.mu, (N,1))
        smpls += np.dot(np.random.randn(N,self.D), self.chol)
        return smpls

class _VectorizedLinearGaussianDynamicalSystemProposal(Proposal):
    # Transition matrix

    def __init__(self, A, sigma):
        self.A = A
        self.D = A.shape[0]
        assert A.shape[1] == self.D, "Transition matrix A must be square!"
        self.sigma = sigma

        self.U_sigma = np.linalg.cholesky(self.sigma).T.copy()

    def predict(self, zpred, z, ts):
        T, N, D = z.shape
        S = ts.shape[0]

        zref = np.asarray(zpred)

        for s in range(S):
            t = ts[s]
            zref[t+1] = zref[t].dot(self.A.T)

    def sample_next(self, z, i_prev, ancestors):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param i_prev:  Time index into z and self.ts

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        N = z.shape[1]
        D = z.shape[2]
        zref = np.asarray(z)

        # Preallocate random variables
        rands = np.random.randn(N,D)

        # Dot with the Cholesky
        rands = np.dot(rands, self.U_sigma)
        zref[i_prev+1] = np.dot(zref[i_prev, ancestors,:], self.A.T) \
                         + rands

    def logp(self, z_prev, i_prev, z_curr, lp):
        """ Compute the log probability of transitioning from z_prev to z_curr
            at time self.ts[i_prev] to self.ts[i_prev+1]

            :param z_prev:  NxD buffer of particle states at the i_prev-th time index
            :param i_prev:  Time index into self.ts
            :param z_curr:  D buffer of particle states at the (i_prev+1)-th time index
            :param lp:      NxM buffer in which to store the probability of each transition

            :return         z[i_prev+1,:,:] is updated with a sample
                            from the proposal distribution.
        """
        z_mean = np.dot(z_prev, self.A.T)
        z_diff = z_curr - z_mean

        lpref = -0.5 * (np.linalg.solve(self.sigma, z_diff.T) * z_diff.T).sum(axis=0)
        np.copyto(np.asarray(lp), lpref)



class _CategoricalLikelihood(Likelihood):
    """
    Multinomial observations from a logistic normal LDS
    """
    def __init__(self, C):

        self.C = C
        self.O = C.shape[0]
        self.D = C.shape[1]

    def logp(self, z, x, i, ll):
        """ Compute the log likelihood, log p(x|z), at time index i and put the
            output in the buffer ll.

            :param z:   TxNxD buffer of latent states
            :param x:   TxO buffer of observations
            :param i:   Time index at which to compute the log likelihood
            :param ll:  N buffer to populate with log likelihoods

            :return     Buffer ll should be populated with the log likelihood of
                        each particle.
        """
        # x_mean is N x O
        x_mean = np.dot(z[i], self.C.T)
        x_pi = np.exp(x_mean)
        x_pi /= x_pi.sum(axis=1)[:,None]

        llref = np.sum(x[i] * np.log(x_pi), axis=1)
        np.copyto(np.asarray(ll), llref)

    def sample(self, z, x, i,n):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param x:       NxD output buffer for observations
            :param i:       Time index to sample
            :param n:       Particle index to sample
        """
        x_mean = np.dot(self.C, z[i,n,:])
        x_pi = np.exp(x_mean)
        x_pi /= np.sum(x_pi)


        from pybasicbayes.util.stats import sample_discrete
        s = sample_discrete(x_pi)
        x[i,:] = 0
        x[i,s] = 1


class LogisticNormalMultinomialLDS(NonstationaryLDS):
    def __init__(self, init_dynamics_distn, dynamics_distn,emission_distn, mu=None, sigma_C=0.01):
        super(LogisticNormalMultinomialLDS, self).__init__(init_dynamics_distn,dynamics_distn, emission_distn)
        self.sigma_C = sigma_C
        if mu is None:
            self.mu = np.zeros(self.p)
        else:
            assert mu.shape == (self.p,)
            self.mu = mu

    def log_likelihood(self):
        ll = 0
        for states in self.states_list:
            psi = states.stateseq.dot(self.C.T) + self.mu
            pi = ln_psi_to_pi(psi)
            ll += np.sum(states.data * np.log(pi))
        return ll

    def add_data(self,data,**kwargs):
        assert isinstance(data,np.ndarray)
        self.states_list.append(LogisticNormalMultinomialLDSStates(model=self,data=data,**kwargs))
        return self

    def generate(self,T,keep=True):
        raise NotImplementedError
        # s = LogisticNormalMultinomialLDSStates(
        #     model=self,T=T,initialize_from_prior=True)
        # data = self._generate_obs(s)
        # if keep:
        #     self.states_list.append(s)
        # return data, s.stateseq

    def _generate_obs(self,s):
        raise NotImplementedError
        # if s.data is None:
        #     s.data = self.emission_distn.rvs(x=s.stateseq,return_xy=False)
        # else:
        #     # filling in missing data
        #     raise NotImplementedError
        # return s.data

    def predictive_log_likelihood(self, Xtest, data_index=0, Npred=100):
        """
        Hacky way of computing the predictive log likelihood
        :param X_pred:
        :param data_index:
        :param M:
        :return:
        """
        Tpred = Xtest.shape[0]

        # Sample particle trajectories
        preds = self.states_list[data_index].sample_predictions(Tpred, Npred)
        preds = np.transpose(preds, [2,0,1])
        assert preds.shape == (Npred, Tpred, self.n)

        psis = np.array([pred.dot(self.C.T) + self.mu for pred in preds])
        pis = np.array([ln_psi_to_pi(psi) for psi in psis])

        # TODO: Generalize for multinomial
        lls = np.zeros(Npred)
        for m in range(Npred):
            # lls[m] = np.sum(
            #     [Multinomial(weights=pis[m,t,:], K=self.p).log_likelihood(Xtest[t][None,:])
            #      for t in xrange(Tpred)])
            lls[m] = np.nansum(Xtest * np.log(pis[m]))


        # Compute the average
        hll = logsumexp(lls) - np.log(Npred)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, Npred), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(Npred)
        std_hll = hll_samples.std()

        return hll, std_hll

    def resample_parameters(self):
        self.resample_init_dynamics_distn()
        self.resample_dynamics_distn()
        self.resample_C()

    def resample_C(self, sigma_prop=0.1, n_steps=10):
        def log_joint_C(C):
            ll = 0
            for states in self.states_list:
                z = states.stateseq
                psi = z.dot(C.T) + self.mu
                pi = ln_psi_to_pi(psi)

                # TODO: Generalize for multinomial
                ll += np.nansum(states.data * np.log(pi))

            ll += (-0.5*C**2/self.sigma_C).sum()

            return ll

        def propose_C(C):
            noise = sigma_prop*np.random.randn(*C.shape)
            return C + noise

        def q(C, Chat):
            # noise = Chat - C
            # return -0.5 * noise ** 2 / sigma_prop**2
            # Spherical proposal = 0! != 1
            return 0

        self.C = mh(self.C, log_joint_C, q, propose_C,
                    steps=n_steps)[-1]



class ParticleSBMultinomialLDSStates(LDSStates):

    def __init__(self, model,T=None,data=None,stateseq=None,
            initialize_from_prior=True):

        super(ParticleSBMultinomialLDSStates, self).\
            __init__(model, T=T, data=data.copy(), stateseq=stateseq,
                     initialize_from_prior=initialize_from_prior)

    @property
    def mu(self):
        return self.model.mu

    @property
    def pi(self):
        psi = self.stateseq.dot(self.C.T) + self.mu
        return psi_to_pi(psi)

    def resample(self, N_particles=100, N_iter=1):
        from hips.inference.particle_mcmc import ParticleGibbsAncestorSampling


        init = _GaussianInitialDistribution(self.mu_init, self.sigma_init)
        prop = _VectorizedLinearGaussianDynamicalSystemProposal(self.A, self.sigma_states)
        lkhd = _CategoricalSBMultinomialLikelihood(self.C, self.mu)

        T, D, z, x = self.T, self.n, self.stateseq, self.data
        pf = ParticleGibbsAncestorSampling(T, N_particles, D)
        pf.initialize(init, prop, lkhd, x, z)

        for i in range(N_iter):
            # Reinitialize with the previous particle
            pf.initialize(init, prop, lkhd, x, z)
            z = np.asarray(pf.sample())

        self.stateseq = z

    def sample_predictions(self, Tpred, Npred, obs_noise=True):
        A, sigma_states, z = self.A, self.sigma_states, self.stateseq
        randseq = np.einsum(
            'tjn,ij->tin',
            np.random.randn(Tpred-1, self.n, Npred),
            np.linalg.cholesky(self.sigma_states))

        states = np.empty((Tpred, self.n, Npred))
        states[0] = np.random.multivariate_normal(A.dot(z[-1]), sigma_states, size=Npred).T
        for t in range(1, Tpred):
            states[t] = self.A.dot(states[t-1]) + randseq[t-1]

        return states


class _CategoricalSBMultinomialLikelihood(Likelihood):
    """
    Multinomial observations from a logistic normal LDS
    """
    def __init__(self, C, mu):

        self.C = C
        self.O = C.shape[0]
        self.D = C.shape[1]
        self.mu = mu

    def logp(self, z, x, i, ll):
        """ Compute the log likelihood, log p(x|z), at time index i and put the
            output in the buffer ll.

            :param z:   TxNxD buffer of latent states
            :param x:   TxO buffer of observations
            :param i:   Time index at which to compute the log likelihood
            :param ll:  N buffer to populate with log likelihoods

            :return     Buffer ll should be populated with the log likelihood of
                        each particle.
        """
        # psi is N x O
        psi = np.dot(z[i], self.C.T) + self.mu
        pi = psi_to_pi(psi)

        llref = np.nansum(x[i] * np.log(pi), axis=1)
        np.copyto(np.asarray(ll), llref)

    def sample(self, z, x, i,n):
        """ Sample the next state given the previous time index

            :param z:       TxNxD buffer of particle states
            :param x:       NxD output buffer for observations
            :param i:       Time index to sample
            :param n:       Particle index to sample
        """
        psi = np.dot(self.C, z[i,n,:]) + self.mu
        pi = psi_to_pi(psi)


        from pybasicbayes.util.stats import sample_discrete
        s = sample_discrete(pi)
        x[i,:] = 0
        x[i,s] = 1


class ParticleSBMultinomialLDS(NonstationaryLDS):
    def __init__(self,init_dynamics_distn, dynamics_distn,emission_distn, mu, sigma_C=0.01):
        super(ParticleSBMultinomialLDS, self).__init__(init_dynamics_distn, dynamics_distn, emission_distn)
        self.mu = mu

        self.sigma_C = sigma_C

    def log_likelihood(self):
        ll = 0
        for states in self.states_list:
            psi = states.stateseq.dot(self.C.T) + self.mu
            pi = psi_to_pi(psi)
            ll += np.sum(states.data * np.log(pi))
        return ll

    def add_data(self,data,**kwargs):
        assert isinstance(data,np.ndarray)
        self.states_list.append(ParticleSBMultinomialLDSStates(model=self,data=data,**kwargs))
        return self

    def generate(self,T,keep=True):
        raise NotImplementedError
        # s = LogisticNormalMultinomialLDSStates(
        #     model=self,T=T,initialize_from_prior=True)
        # data = self._generate_obs(s)
        # if keep:
        #     self.states_list.append(s)
        # return data, s.stateseq

    def _generate_obs(self,s):
        raise NotImplementedError
        # if s.data is None:
        #     s.data = self.emission_distn.rvs(x=s.stateseq,return_xy=False)
        # else:
        #     # filling in missing data
        #     raise NotImplementedError
        # return s.data

    def predictive_log_likelihood(self, Xtest, data_index=0, Npred=100):
        """
        Hacky way of computing the predictive log likelihood
        :param X_pred:
        :param data_index:
        :param M:
        :return:
        """
        Tpred = Xtest.shape[0]

        # Sample particle trajectories
        preds = self.states_list[data_index].sample_predictions(Tpred, Npred)
        preds = np.transpose(preds, [2,0,1])
        assert preds.shape == (Npred, Tpred, self.n)

        psis = np.array([pred.dot(self.C.T) + self.mu for pred in preds])
        pis = np.array([psi_to_pi(psi) for psi in psis])

        # TODO: Generalize for multinomial
        lls = np.zeros(Npred)
        for m in range(Npred):
            # lls[m] = np.sum(
            #     [Multinomial(weights=pis[m,t,:], K=self.p).log_likelihood(Xtest[t][None,:])
            #      for t in xrange(Tpred)])
            lls[m] = np.nansum(Xtest * np.log(pis[m]))


        # Compute the average
        hll = logsumexp(lls) - np.log(Npred)

        # Use bootstrap to compute error bars
        samples = np.random.choice(lls, size=(100, Npred), replace=True)
        hll_samples = logsumexp(samples, axis=1) - np.log(Npred)
        std_hll = hll_samples.std()

        return hll, std_hll

    def resample_parameters(self):
        self.resample_init_dynamics_distn()
        self.resample_dynamics_distn()
        self.resample_C()

    def resample_C(self, sigma_prop=0.01, n_steps=10):
        def log_joint_C(C):
            ll = 0
            for states in self.states_list:
                z = states.stateseq
                psi = z.dot(C.T) + self.mu
                pi = psi_to_pi(psi)

                # TODO: Generalize for multinomial
                ll += np.nansum(states.data * np.log(pi))

            ll += (-0.5*C**2/self.sigma_C).sum()

            return ll

        def propose_C(C):
            noise = sigma_prop*np.random.randn(*C.shape)
            return C + noise

        def q(C, Chat):
            # noise = Chat - C
            # return -0.5 * noise ** 2 / sigma_prop**2
            # Spherical proposal = 0! != 1
            return 0

        self.C = mh(self.C, log_joint_C, q, propose_C,
                    steps=n_steps)[-1]
