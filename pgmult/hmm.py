"""
Super simple class to wrap an HMM with multinomial observations
"""
import numpy as np
from scipy.special import gammaln

from pyhsmm.models import HMM
from pybasicbayes.distributions import Multinomial


class MultinomialHMM(HMM):
    def __init__(self, K, D,
                 alpha_0=1,                     # Concentration for multinomial obs weights
                 alpha_trans=1,                 # Concentration for multinomial obs weights
                 alpha_a_0=None,                # Prior for transition concentration
                 alpha_b_0=None,                # Prior for transition concentration
                 init_state_concentration=1     # Initial state concentration
                 ):
        """
        Initialize an HMM with D latent states and
        K dimensional Multinomial observations.
        """
        obs_distns = [Multinomial(K=K, alpha_0=alpha_0) for _ in range(D)]

        super(HMM, self).__init__(
            obs_distns,
            alpha=alpha_trans, alpha_a_0=alpha_a_0, alpha_b_0=alpha_b_0,
            init_state_concentration=init_state_concentration)

    @property
    def pis(self):
        return [np.array([self.obs_distns[s].weights
                          for s in seq])
                for seq in self.stateseqs]

    def log_likelihood_fixed_z(self):
        # We generally want to compute the log likelihood
        pis = self.pis
        Xs = [s.data for s in self.states_list]
        ll = 0
        for pi,X in zip(pis, Xs):
            ll += gammaln(X.sum(axis=1) + 1).sum() - gammaln(X+1).sum()
            ll += (X * np.log(pi)).sum()
        return ll

    # TODO: Override resample_obs_distns with a Cython function to aggregate
    # TODO: the sufficient statistics of the multinomial observations.
