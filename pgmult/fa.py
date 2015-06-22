from __future__ import division
import sys
import numpy as np
from scipy.linalg import solve, solve_triangular
from scipy.linalg.lapack import dpotrs

from pybasicbayes.abstractions import Model, ModelGibbsSampling
from pybasicbayes.distributions import Gaussian, Regression
from autoregressive.distributions import AutoRegression

from lds import MultinomialLDS
from distributions import PGMultinomialRegression
from internals.utils import psi_to_pi, solve_diagonal_plus_lowrank


class LazyMultinomialFA(MultinomialLDS):
    """
    The lazy way to implement factor analysis is as a
    special case of a linear dynamical system
    """
    def __init__(self,K,n,C=None,sigma_C=1.0):
        dynamics_distn = AutoRegression(A=np.zeros((n,n)),sigma=np.eye(n))
        init_dynamics_distn = Gaussian(mu=np.zeros(n),sigma=np.eye(n))
        super(LazyMultinomialFA,self).__init__(
            K,n, dynamics_distn, init_dynamics_distn, C=C, sigma_C=sigma_C)

    def resample_dynamics_distn(self):
        pass

    def resample_init_dynamics_distn(self):
        pass


class _MultinomialFABase(Model):
    """
    The not super hacky way of doing factor analysis.
    This should be a little faster if we had an efficient
    implementation.
    """
    def __init__(self, K, n, C=None, sigma_C=None,
                 mu=None, mu_pi=None):
        self.K, self.n = K, n
        self.emission_distn = PGMultinomialRegression(
            K, n, C=C, sigma_C=sigma_C, mu=mu, mu_pi=mu_pi)
        self.data_list = []

    @property
    def C(self):
        return self.emission_distn.C

    def psi(self, augmented_data):
        psi = self.emission_distn.psi(augmented_data)
        return psi

    def pi(self, augmented_data):
        psi = self.psi(augmented_data)
        return np.array([psi_to_pi(p) for p in psi])

    def add_data(self, data, z=None):
        assert isinstance(data, np.ndarray) and data.ndim == 2 \
               and data.shape[1] == self.K and data.min() >= 0
        M = data.shape[0]

        # Create a new factor for the data
        if z is None:
            z = np.random.randn(M, self.n)
        else:
            assert z.shape == (M, self.n)

        augmented_data = {"M": M,
                          "T": M,
                          "x": data,
                          "z": z}

        # The emission distribution handles the omegas
        self.emission_distn.augment_data(augmented_data)

        self.data_list.append(augmented_data)
        return augmented_data

    def log_likelihood(self):
        ll = 0
        for data in self.data_list:
            ll += self.emission_distn.log_likelihood(data)

        return ll

    def generate(self,keep=True, M=1, N=1, full_output=False):
        """

        :param keep:
        :param M:
        :param N:
        :return:
        """
        z = np.random.randn(M, self.n)
        x = self.emission_distn.rvs(z, N=N)

        # Package into an augmented data dict
        augmented_data = {"M": M,
                          "T": M,
                          "x": x,
                          "z": z}

        if full_output:
            return augmented_data
        else:
            return x


class _MultinomialFAGibbs(_MultinomialFABase, ModelGibbsSampling):
    def resample_model(self):
        self.resample_factors()
        self.emission_distn.resample(self.data_list)

    def _naive_compute_z_posterior(self, omegam, kappam, mu):
        # Compute the posterior parameters
        # This can be done more efficiently since it is low rank plus diagonal
        # see below
        C, n = self.C, self.n

        sigma_psi = C.dot(C.T) + np.diag(1./omegam)
        mu_z_post = C.T.dot(solve(sigma_psi, kappam/omegam - mu))
        sigma_z_post = np.eye(self.n) - C.T.dot(solve(sigma_psi, C))

        return mu_z_post, sigma_z_post

    def _compute_z_posterior_given_omega(self, omegam, kappam, mu):
        # Compute the posterior parameters
        # This uses the matrix inversion lemma
        C, n = self.C, self.n

        # Omegam = np.diag(omegam)
        # sigma_z_post = np.linalg.inv(np.eye(n) + C.T.dot(Omegam).dot(C))
        # COC = np.einsum("wi,wj,w->ij", C,C,omegam)
        COC = (C * omegam[:,None]).T.dot(C)
        CO = (C * omegam[:,None])
        L_z_post = np.linalg.cholesky(np.eye(n) + COC)
        # mu_z_post = sigma_z_post.dot(C.T).dot(Omegam).dot(kappam/omegam)
        mu_z_post = dpotrs(L_z_post, CO.T.dot(kappam/omegam - mu), lower=True)[0]

        return mu_z_post, L_z_post

    def _compute_z_posterior(self, mean, prec):
        # Compute the posterior parameters
        # This uses the matrix inversion lemma
        C, n = self.C, self.n

        # Omegam = np.diag(omegam)
        # sigma_z_post = np.linalg.inv(np.eye(n) + C.T.dot(Omegam).dot(C))
        # COC = np.einsum("wi,wj,w->ij", C,C,omegam)
        COC = (C * prec[:,None]).T.dot(C)
        CO = (C * prec[:,None])
        L_z_post = np.linalg.cholesky(np.eye(n) + COC)
        # mu_z_post = sigma_z_post.dot(C.T).dot(Omegam).dot(kappam/omegam)
        mu_z_post = dpotrs(L_z_post, CO.T.dot(mean), lower=True)[0]

        return mu_z_post, L_z_post

    def resample_factors(self):
        mu = self.emission_distn.mu

        for data in self.data_list:

            conditional_mean = self.emission_distn.conditional_mean(data)
            conditional_prec = self.emission_distn.conditional_prec(data, flat=True)

            for m in xrange(data["M"]):
                # omegam = data["omega"][m]
                # # TODO: Handle this in resample omega
                # omegam[omegam==0] = 1e-32
                # kappam = data["kappa"][m]
                # mu_z_post, L_z_post = \
                #     self._compute_z_posterior(omegam, kappam, mu)
                mu_z_post, L_z_post = \
                    self._compute_z_posterior(conditional_mean[m],
                                              conditional_prec[m])
                # Check against the naive computation
                # mu_z_post_naive, sigma_z_post_naive = \
                #     self._naive_compute_z_posterior(omegam, kappam)
                # assert np.allclose(mu_z_post, mu_z_post_naive)
                # assert np.allclose(sigma_z_post, sigma_z_post_naive)

                # data["z"][m] = \
                #     np.random.multivariate_normal(mu_z_post, sigma_z_post)
                rand_vec = np.random.randn(self.n)
                data["z"][m] = mu_z_post + solve_triangular(L_z_post,  rand_vec, lower=True, trans='T')

class MultinomialFA(_MultinomialFAGibbs):
    pass


class MixedFARegression(object):
    # TODO diagonal regression class

    def __init__(self,xs,u,y,Nlatent,z=None,sigma_C=1.,zu_hyps={},zy_hyps={},zu_affine=True,zy_affine=True):
        Nmult = len(xs)
        Ks = [x.shape[1] for x in xs]
        Ninput, Nreal = u.shape
        Ninput, Nout = y.shape
        self.Nlatent, self.Ninput, self.Nmult, self.Nreal, Ks \
            = Nlatent, Ninput, Nmult, Nreal, Ks

        self.xs = xs
        self.u = u
        self.y = y
        self.z = np.random.randn(Ninput,Nlatent) if z is None else z

        default_zu_hyps = dict(
            nu_0=Nreal+1, S_0=np.eye(Nreal), M_0=np.zeros((Nreal,Nlatent+zu_affine)),
            K_0=np.eye(Nlatent+zu_affine), affine=zu_affine)
        self.zu_regression = Regression(**dict(default_zu_hyps,**zu_hyps))

        default_zy_hyps = dict(
            nu_0=Nout+1, S_0=np.eye(Nout), M_0=np.zeros((Nout,Nlatent+zy_affine)),
            K_0=np.eye(Nlatent+zy_affine), affine=zy_affine)
        self.zy_regression = Regression(**dict(default_zy_hyps,**zy_hyps))

        # Initialize zx regressions with mean offset
        from pgmult.internals.utils import compute_psi_cmoments
        mus = [compute_psi_cmoments(np.ones(K))[0] for K in Ks]
        self.zx_regressions = \
            [PGMultinomialRegression(K,Nlatent,mu=mu,sigma_C=sigma_C) for K, mu in zip(Ks, mus)]
        self.data_list = [dict(T=Ninput,z=self.z, x=x) for x in self.xs]
        for data, regr in zip(self.data_list, self.zx_regressions):
            regr.augment_data(data)

    def log_likelihood(self):
        ll = 0
        ll += self.zu_regression.log_likelihood(np.hstack((self.z, self.u))).sum()
        for data, regr in zip(self.data_list, self.zx_regressions):
            ll += regr.log_likelihood(data)

        yvalid = np.all(np.isfinite(self.y), axis=1)
        ll += self.zy_regression.log_likelihood(np.hstack((self.z[yvalid], self.y[yvalid]))).sum()
        return ll

    def predict(self):
        if not self.zy_regression.affine:
            return self.z.dot(self.A.T)
        else:
            A, b = self.zy_regression.A[:,:-1], self.zy_regression.A[:,-1]
            return self.z.dot(A.T) + b

    def resample_model(self):
        self.resample_zy_regression()
        self.resample_zu_regression()
        self.resample_zx_regressions()
        # TODO: Understand why the Gibbs sampler fails when z is resampled first
        self.resample_z()

    def resample_zy_regression(self):
        self.zy_regression.resample(np.hstack((self.z,self.y)))

    def resample_zu_regression(self):
        self.zu_regression.resample(np.hstack((self.z,self.u)))

    def resample_zx_regressions(self):
        for regression, augdata in zip(self.zx_regressions,self.data_list):
            regression.resample([augdata])

    def resample_z(self):
        # TODO this is written for a single z, not Ninput of them
        for n in xrange(self.Ninput):
            mu, Sigma = self._gaussian_condition_on(
                self._B(n), self._d(n), self._condvals(n))
            self.z[n] = np.random.multivariate_normal(mu, Sigma)

    def _B(self, n):
        if np.isnan(self.y[n]).any():
            return np.vstack([self.C, self.Cu])
        else:
            return np.vstack([self.C, self.Cu, self.A])

    def _d(self, n):
        if np.isnan(self.y[n]).any():
            return np.concatenate([1./self.omegas(n), self.sigmas_u])
        else:
            return np.concatenate([1./self.omegas(n), self.sigmas_u, self.sigmas_y])

    def _condvals(self, n):
        # TODO this could be faster
        psi_vals = \
            [regr.conditional_mean(data)[n]
             for regr, data in zip(self.zx_regressions, self.data_list)]
        return np.concatenate(
            psi_vals
            + [self.u[n] - self.bu]
            + ([self.y[n] - self.b] if not np.isnan(self.y[n]).any() else []))

    def _gaussian_condition_on(self,B,d,xb):
        solves = solve_diagonal_plus_lowrank(d,B,B.T,np.hstack([xb[:,None],B]))
        mu = B.T.dot(solves[:,0])
        Sigma = np.eye(self.Nlatent) - B.T.dot(solves[:,1:])
        return mu, Sigma

    @property
    def C(self):
        if self.Nmult > 0:
            return np.vstack([regr.C for regr in self.zx_regressions])
        else:
            return np.zeros((0, self.Nlatent))

    @property
    def Cu(self):
        return self.zu_regression.A if not self.zu_regression.affine \
            else self.zu_regression.A[:,:-1]

    @property
    def bu(self):
        return 0. if not self.zu_regression.affine \
            else self.zu_regression.A[:,-1]

    @property
    def A(self):
        return self.zy_regression.A if not self.zy_regression.affine \
            else self.zy_regression.A[:,:-1]

    @property
    def b(self):
        return 0. if not self.zy_regression.affine \
            else self.zy_regression.A[:,-1]

    def omegas(self, n):
        if self.Nmult > 0:
            return np.hstack([data['omega'][n] for data in self.data_list])
        else:
            return np.array([])

    @property
    def sigmas_y(self):
        # TODO: this should be diagonal of a covariance matrix
        return np.diag(self.zy_regression.sigma)

    @property
    def sigmas_u(self):
        # TODO: this should be diagonal of a covariance matrix
        return np.diag(self.zu_regression.sigma)


