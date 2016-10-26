"""
Multinomial observations with Gaussian process priors.
"""

import sys
import numpy as np

from scipy.linalg import solve_triangular
from scipy.linalg.lapack import dpotrs
from scipy.special import gammaln

from pybasicbayes.abstractions import Model, ModelGibbsSampling

from pgmult.utils import \
    kappa_vec, N_vec, psi_to_pi, pi_to_psi, ln_psi_to_pi, ln_pi_to_psi, \
    compute_uniform_mean_psi, initialize_polya_gamma_samplers

import pypolyagamma as ppg

from hips.inference.elliptical_slice import elliptical_slice

#TODO: Factor out reusable parts of the GP models
#TODO: GPData class

class _MultinomialGPBase(Model):
    def __init__(self, K, kernel, D, Z=None, X=None, mu=None, mu_0=None, Sigma_0=None):
        self.K = K
        self.D = D
        self.kernel = kernel

        # Start with an empty datalist
        self.data_list = []

        # Initialize the GP mean prior
        if not any([_ is None for _ in (mu_0, Sigma_0)]):
            assert mu_0.shape == (self.K-1,)
            assert Sigma_0.shape == (self.K-1, self.K-1)
            self.mu_0, self.Sigma_0 = mu_0, Sigma_0
        else:
            self.mu_0 = compute_uniform_mean_psi(K)[0]
            self.Sigma_0 = np.eye(self.K-1)

        # Initialize the GP mean
        if mu is None:
            self.mu = self.mu_0.copy()
        else:
            assert isinstance(mu, np.ndarray) and mu.shape == (self.K-1,)
            self.mu = mu

    def psi(self, augmented_data):
        return augmented_data["psi"] + self.mu[None, :]

    def pi(self, augmented_data):
        psi = self.psi(augmented_data)
        pi = psi_to_pi(psi)

        return pi

    def add_data(self, Z, X, fixed_kernel=True):
        # Z is the array of points where multinomial vectors are observed
        # X is the corresponding set of multinomial vectors
        assert Z.ndim == 2 and Z.shape[1] == self.D
        M = Z.shape[0]
        assert X.shape == (M, self.K), "X must be MxK"

        # Compute kappa and N for each of the m inputs
        N = N_vec(X).astype(np.float)
        kappa = kappa_vec(X)

        # Initialize the auxiliary variables
        omega = np.ones((M, self.K-1))

        # Initialize a "sample" from psi
        psi = np.zeros((M, self.K-1))

        # Precompute the kernel for the case where it is fixed
        if fixed_kernel:
            C = self.kernel.K(Z)
            C += 1e-6 * np.eye(M)
            C_inv = np.linalg.inv(C)
        else:
            C = None
            C_inv = None

        # Pack all this up into a dict
        augmented_data = \
        {
            "X":        X,
            "Z":        Z,
            "M":        M,
            "N":        N,
            "C":        C,
            "C_inv":    C_inv,
            "kappa":    kappa,
            "omega":    omega,
            "psi":      psi
        }
        self.data_list.append(augmented_data)
        return augmented_data

    def log_likelihood(self):
        """
        Compute the log likelihood of the observed
        :return:
        """
        ll = 0
        for data in self.data_list:
            x = data["X"]

            # Compute \pi for the latent \psi
            pi = self.pi(data)

            # Add the multinomial likelihood given \pi
            ll += gammaln(x.sum(axis=1)+1).sum() - gammaln(x+1).sum()
            ll += np.nansum(x * np.log(pi))

        return ll

    def predictive_log_likelihood(self, Z_pred, X_pred):
        """
        Predict the GP value at the inputs Z_pred and evaluate the likelihood of X_pred
        """
        _, mu_pred, Sig_pred = self.collapsed_predict(Z_pred, full_output=True)

        psis = np.array([np.random.multivariate_normal(mu, Sig) for mu,Sig in zip(mu_pred, Sig_pred)])
        pis = psi_to_pi(psis.T)

        pll = 0
        pll += gammaln(X_pred.sum(axis=1)+1).sum() - gammaln(X_pred+1).sum()
        pll += np.nansum(X_pred * np.log(pis))

        return pll, pis


    def generate(self, keep=True, Z=None, N=None, full_output=True):
        assert Z is not None and Z.ndim == 2 and Z.shape[1] == self.D
        M = Z.shape[0]

        assert N.ndim == 1 and N.shape[0] == M and np.all(N) >= 1
        assert N.dtype in (np.int32, np.int)
        N = N.astype(np.int32)

        # Compute the covariance of the Z's
        C = self.kernel.K(Z)

        # Sample from a zero mean GP, N(0, C) for each output, k
        psis = np.zeros((M, self.K-1))
        for k in range(self.K-1):
            # TODO: Reuse the Cholesky
            psis[:,k] = np.random.multivariate_normal(np.zeros(M), C)

        # Add the mean vector
        psis += self.mu[None,:]

        # Sample from the multinomial distribution
        pis = psi_to_pi(psis)
        X = np.array([np.random.multinomial(N[m], pis[m]) for m in range(M)])

        if keep:
            self.add_data(Z, X)

        if full_output:
            return X, psis
        else:
            return X

    def rvs(self, Z):
        return self.generate(keep=False, Z=Z)

    def copy_sample(self):
        # Return psi, omega, and mu
        return self.mu.copy(), [(data["psi"].copy(), data["omega"].copy()) for data in self.data_list]

    def set_sample(self, sample):
        mu, psi_omega_list = sample
        self.mu = mu
        assert len(psi_omega_list) == len(self.data_list)
        for (psi, omega), data in zip(psi_omega_list, self.data_list):
            data["psi"] = psi
            data["omega"] = omega

    def collapsed_predict(self, Z_new, full_output=True, full_cov=False):
        """
        Predict the multinomial probability vector at a grid of points, Z_new
        by first integrating out the value of psi at the data, Z_test, given
        omega and the kernel parameters.
        """
        assert len(self.data_list) == 1, "Must have one data list in order to predict."
        data = self.data_list[0]
        Z = data["Z"]

        assert Z_new is not None and Z_new.ndim == 2 and Z_new.shape[1] == self.D
        M_new = Z_new.shape[0]

        # Compute the kernel for Z_news
        C   = self.kernel.K(Z, Z)
        Cnn = self.kernel.K(Z_new, Z_new)
        Cnv = self.kernel.K(Z_new, Z)

        # Predict the psis
        mu_psis_new = np.zeros((self.K-1, M_new))
        Sig_psis_new = np.zeros((self.K-1, M_new, M_new))
        for k in range(self.K-1):
            sys.stdout.write(".")
            sys.stdout.flush()

            # Throw out inputs where N[:,k] == 0
            Omegak = data["omega"][:,k]
            kappak = data["kappa"][:,k]

            # Set the precision for invalid points to zero
            Omegak[Omegak == 0] = 1e-16

            # Account for the mean from the omega potentials
            y = kappak/Omegak - self.mu[k]

            # The y's are noisy observations at inputs Z
            # with diagonal covariace Omegak^{-1}
            Cvv_noisy = C + np.diag(1./Omegak)
            Lvv_noisy = np.linalg.cholesky(Cvv_noisy)

            # Compute the conditional mean given noisy observations
            psik_pred = Cnv.dot(dpotrs(Lvv_noisy, y, lower=True)[0])

            # Save these into the combined arrays
            mu_psis_new[k] = psik_pred + self.mu[k]

            if full_cov:
                Sig_psis_new[k] = Cnn - Cnv.dot(dpotrs(Lvv_noisy, Cnv.T, lower=True)[0])

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Convert these to pis
        pis_new = psi_to_pi(mu_psis_new)

        if full_output:
            return pis_new, mu_psis_new, Sig_psis_new
        else:
            return pis_new

    def marginal_likelihood(self, data):
        """
        Compute marginal likelihood with the given values of omega
        :param augmented_data:
        :return:
        """
        mll = 0

        M = data["M"]
        Z = data["Z"]
        omega = data["omega"]
        kappa = data["kappa"]

        # Compute the kernel for Z_news
        Cvv  = self.kernel.K(Z, Z)
        Cvv +=  np.diag(1e-6 * np.ones(M))

        for k in range(self.K-1):
            sys.stdout.write(".")
            sys.stdout.flush()

            # The predictive mean and covariance are given by equations 2.22-2.24
            # of GPML by Rasmussen and Williams
            Cvvpo = Cvv + np.diag(omega[:,k])
            Lvvpo = np.linalg.cholesky(Cvvpo)

            # The "observations" are the effective mean of the Gaussian
            # likelihood given omega
            y = kappa[:,k] / omega[:,k] - self.mu[k]

            # First term is -1/2 y^T K_y^{-1} y
            x1 = solve_triangular(Lvvpo, y, lower=True)
            x2 = solve_triangular(Lvvpo.T, x1, lower=False)
            mll += -0.5 * y.T.dot(x2)

            # Second term: -1/2 log |K_y|
            mll += np.linalg.slogdet(Cvvpo)[1]

            # Third term: -M/2 log 2*pi
            mll += -M/2.0 * np.log(2*np.pi)

        return mll

    def grad_marg_likelihood(self, data):
        """
        Compute the gradient of the marginal likelihood for a given dataset wrt K
        This is given by GPML equation 5.9
        :return:
        """
        dL_dK = 0

        M = data["M"]
        Z = data["Z"]
        omega = data["omega"]
        kappa = data["kappa"]

        # Compute the kernel for Z_news
        Cvv  = self.kernel.K(Z, Z)
        Cvv +=  np.diag(1e-6 * np.ones(M))

        for k in range(self.K-1):
            sys.stdout.write(".")
            sys.stdout.flush()

            # The predictive mean and covariance are given by equations 2.22-2.24
            # of GPML by Rasmussen and Williams
            Cvvpo = Cvv + np.diag(omega[:,k])
            iCvvpo = np.linalg.inv(Cvvpo)

            # The "observations" are the effective mean of the Gaussian
            # likelihood given omega
            y = kappa[:,k] / omega[:,k] - self.mu[k]

            # Compute alpha in Eqn 5.9
            # x1 = solve_triangular(Lvvpo, y, lower=True)
            # alpha = solve_triangular(Lvvpo.T, x1, lower=False)
            alpha = iCvvpo.dot(y)

            # The gradient wrt theta is given by
            #   Tr(dL_dK * dK_dtheta)
            # = Tr(1/2 * (alpha * alpha^T - K^{-1}) * dK/dtheta)
            # where theta is a param of the kernel.
            # Here we just compute the first term, dL_dK
            dL_dK += 0.5 * (np.outer(alpha, alpha) - iCvvpo)

        return dL_dK

class _MultinomialGPGibbsSampling(_MultinomialGPBase, ModelGibbsSampling):

    def __init__(self,  K, kernel, D, Z=None, X=None, mu=None, mu_0=None, Sigma_0=None):
        super(_MultinomialGPGibbsSampling, self).__init__(K, kernel, D, Z, X, mu=mu, mu_0=mu_0, Sigma_0=Sigma_0)
        self.ppgs = initialize_polya_gamma_samplers()

    def initialize_from_data(self, pi_lim=None, initialize_to_mle=True):
        """
        Initialize the psi's to the empirical mean of the data
        :return:
        """
        for data in self.data_list:
            # Compute the empirical probability
            X = data["X"]
            M = data["M"]
            assert X.ndim == 2 and X.shape[1] == self.K

            # Get the empirical probabilities (offset by 1 to ensure nonzero)
            alpha = 1
            pi_emp = (alpha+X).astype(np.float) / \
                           (alpha + X).sum(axis=1)[:,None]
            pi_emp_mean = pi_emp.mean(axis=0)


            # Set mu equal to the empirical mean value of pi
            psi_emp_mean = pi_to_psi(pi_emp_mean)
            self.mu = psi_emp_mean
            # self.mu = np.zeros(self.K-1)

            if initialize_to_mle:
                # Convert empirical values to psi
                psi_emp = np.array([pi_to_psi(p) for p in pi_emp])
                psi_emp -= self.mu
                assert psi_emp.shape == (M, self.K-1)

                # Set the deviations from the mean to zero
                data["psi"] = psi_emp
            else:
                data["psi"] = np.zeros((M, self.K-1))

        # Resample the omegas
        self.resample_omega()


    def resample_model(self, verbose=False):
        self.resample_psi(verbose=verbose)
        # self.resample_mu()
        self.resample_omega()
        self.resample_kernel_parameters()

    def resample_psi(self, verbose=False):
        for data in self.data_list:
            # import pdb; pdb.set_trace()
            M = data["M"]
            Z = data["Z"]

            # Invert once for all k
            if "C_inv" in data:
                C_inv = data["C_inv"]
            else:
                C = self.kernel.K(Z)
                C += 1e-6 * np.eye(M)
                C_inv = np.linalg.inv(C)

            # Compute the posterior covariance
            psi = np.zeros((M, self.K-1))
            for k in range(self.K-1):
                if verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                # Throw out inputs where N[:,k] == 0
                Omegak = data["omega"][:,k]
                kappak = data["kappa"][:,k]

                # Set the precision for invalid points to zero
                Omegak[Omegak == 0] = 1e-32

                # Account for the mean
                lkhd_mean = kappak/Omegak - self.mu[k]

                # Compute the posterior parameters
                L_post = np.linalg.cholesky(C_inv + np.diag(Omegak))
                mu_post = dpotrs(L_post, Omegak * lkhd_mean, lower=True)[0]

                # Go through each GP and resample psi given the likelihood
                rand_vec = np.random.randn(M)
                psi[:,k] = mu_post + solve_triangular(L_post, rand_vec, lower=True, trans='T')

                assert np.all(np.isfinite(psi[:,k]))

            if verbose:
                sys.stdout.write("\n")
                sys.stdout.flush()

            data["psi"] = psi

    def resample_omega(self):
        # Resample the omega's given N and psi
        for data in self.data_list:
            M = data["M"]
            N = data["N"]
            psi = data["psi"] + self.mu[None, :]

            # Go through each GP and resample psi given the likelihood
            tmp_omg = np.zeros(N.size)
            ppg.pgdrawvpar(self.ppgs, N.ravel(), psi.ravel(), tmp_omg)
            data["omega"] = tmp_omg.reshape((M, self.K-1))

    def resample_mu(self):
        # Resample the mean vector given the samples of psi
        # Work with the precision formulation
        post_prec = 1./np.diag(self.Sigma_0)
        post_prec_dot_mean = 1./np.diag(self.Sigma_0) * self.mu_0

        # Add terms from the data
        for data in self.data_list:
            N, psi, omega, kappa = \
                data["N"], data["psi"], data["omega"], data["kappa"]

            # Compute the residual
            lkhd_mean = kappa/omega - psi

            # Update the sufficient statistics
            for k in range(self.K-1):
                valid = np.where(N[:,k])[0]
                post_prec += omega[valid, k].sum(0)
                post_prec_dot_mean += (omega[valid,k] * lkhd_mean[valid,k]).sum(0)

        # Resample
        post_mean = post_prec_dot_mean / post_prec
        self.mu = post_mean + 1./np.sqrt(post_prec) * np.random.randn(self.K-1)

    def _resample_kernel_parameters_hmc(self):
        """
        Resample the kernel parameters using HMC
        :return:
        """
        import ipdb; ipdb.set_trace()
        assert len(self.data_list) == 1, \
            "Only supporting one data set right now. " \
            "The problem is that the kernel computes the gradient for " \
            "a given set of data points."
        data = self.data_list[0]

        # First compute dL_dK
        dL_dK = self.grad_marg_likelihood(data)

        # Now compute the gradients of the kernel hypers
        self.kernel.update_gradients_full(dL_dK, data["Z"])

        # TODO: Sample new kernel hypers with HMC
        raise NotImplementedError()

    def resample_kernel_parameters(self):
        pass


class MultinomialGP(_MultinomialGPGibbsSampling):
    pass


class LogisticNormalGP(ModelGibbsSampling):
    """
    Alternative model with a logistic normal link function. This is not conjugate
    with the Gaussian prior, so instead we use elliptical slice sampling to
    sample the underlying Gaussian process.
    """
    def __init__(self, K, kernel, D, mu=None, mu_0=None, Sigma_0=None):
        self.K = K
        self.D = D
        self.kernel = kernel

        # Start with an empty datalist
        self.data_list = []

        # Initialize the GP mean prior
        if not any([_ is None for _ in (mu_0, Sigma_0)]):
            assert mu_0.shape == (self.K,)
            assert Sigma_0.shape == (self.K, self.K)
            self.mu_0, self.Sigma_0 = mu_0, Sigma_0
        else:
            self.mu_0 = np.zeros(self.K)
            self.Sigma_0 = np.eye(self.K-1)

        # Initialize the GP mean
        if mu is None:
            self.mu = self.mu_0.copy()
        else:
            assert isinstance(mu, np.ndarray) and mu.shape == (self.K,)
            self.mu = mu

    def psi(self, augmented_data):
        return augmented_data["psi"] + self.mu[None, :]

    def pi(self, augmented_data):
        psi = self.psi(augmented_data)
        return ln_psi_to_pi(psi)

    def add_data(self, Z, X, fixed_kernel=True):
        # Z is the array of points where multinomial vectors are observed
        # X is the corresponding set of multinomial vectors
        # M is the number of datapoints
        assert Z.ndim == 2 and Z.shape[1] == self.D
        M = Z.shape[0]
        assert X.shape == (M, self.K), "X must be MxK"

        # Initialize a "sample" from psi
        psi = np.zeros((M, self.K))

        # Precompute the kernel for the case where it is fixed
        if fixed_kernel:
            C = self.kernel.K(Z)
            C += 1e-6 * np.eye(M)
            C_inv = np.linalg.inv(C)
            L = np.linalg.cholesky(C)
        else:
            C = None
            C_inv = None
            L = None

        # Pack all this up into a dict
        augmented_data = \
        {
            "X":        X,
            "Z":        Z,
            "M":        M,
            "C":        C,
            "C_inv":    C_inv,
            "L":        L,
            "psi":      psi
        }
        self.data_list.append(augmented_data)
        return augmented_data

    def initialize_from_data(self, pi_lim=None, initialize_to_mle=True):
        """
        Initialize the psi's to the empirical mean of the data
        :return:
        """
        for data in self.data_list:
            # Compute the empirical probability
            X = data["X"]
            M = data["M"]
            assert X.ndim == 2 and X.shape[1] == self.K

            # Get the empirical probabilities (offset by 1 to ensure nonzero)
            alpha = 1.0
            pi_emp = (alpha+X).astype(np.float) / \
                           (alpha + X).sum(axis=1)[:,None]
            pi_emp_mean = pi_emp.mean(axis=0)

            # Set mu equal to the empirical mean value of pi
            psi_emp_mean = ln_pi_to_psi(pi_emp_mean)
            # self.mu = psi_emp_mean
            self.mu = np.zeros(self.K)

            if initialize_to_mle:
                # Convert empirical values to psi
                psi_emp = np.array([ln_pi_to_psi(p) for p in pi_emp])
                psi_emp -= self.mu
                assert psi_emp.shape == (M, self.K)

                # Set the deviations from the mean to zero
                data["psi"] = psi_emp
            else:
                data["psi"] = np.zeros((M, self.K))

    def log_likelihood(self, augmented_data=None):
        """
        Compute the log likelihood of the observed
        :return:
        """
        ll = 0
        if augmented_data is None:
            datas = self.data_list
        else:
            datas = [augmented_data]

        for data in datas:
            x = data["X"]

            # Compute \pi for the latent \psi
            pi = self.pi(data)

            # Add the multinomial likelihood given \pi
            ll += gammaln(x.sum(axis=1)+1).sum() - gammaln(x+1).sum()
            ll += np.nansum(x * np.log(pi))

        return ll

    def predictive_log_likelihood(self, Z_pred, X_pred):
        """
        Predict the GP value at the inputs Z_pred and evaluate the likelihood of X_pred
        """
        _, mu_pred, Sig_pred = self.predict(Z_pred, full_output=True)

        psis = np.array([np.random.multivariate_normal(mu, Sig) for mu,Sig in zip(mu_pred, Sig_pred)])
        pis = ln_psi_to_pi(psis.T)

        pll = 0
        pll += gammaln(X_pred.sum(axis=1)+1).sum() - gammaln(X_pred+1).sum()
        pll += np.nansum(X_pred * np.log(pis))

        return pll, pis


    def generate(self, keep=True, Z=None, N=None, full_output=True):
        assert Z is not None and Z.ndim == 2 and Z.shape[1] == self.D
        M = Z.shape[0]

        assert N.ndim == 1 and N.shape[0] == M and np.all(N) >= 1
        assert N.dtype in (np.int32, np.int)
        N = N.astype(np.int32)

        # Compute the covariance of the Z's
        C = self.kernel.K(Z)

        # Sample from a zero mean GP, N(0, C) for each output, k
        psis = np.zeros((M, self.K))
        for k in range(self.K):
            # TODO: Reuse the Cholesky
            psis[:,k] = np.random.multivariate_normal(np.zeros(M), C)

        # Add the mean vector
        psis += self.mu[None,:]

        # Sample from the multinomial distribution
        pis = np.array([ln_psi_to_pi(psi) for psi in psis])
        X = np.array([np.random.multinomial(N[m], pis[m]) for m in range(M)])

        if keep:
            self.add_data(Z, X)

        if full_output:
            return X, psis
        else:
            return X

    def predict(self, Z_new, full_output=True, full_cov=False):
        """
        Predict the multinomial probability vector at a grid of points, Z
        :param Z_new:
        :return:
        """
        assert len(self.data_list) == 1, "Must have one data list in order to predict."
        data = self.data_list[0]
        M = data["M"]
        Z = data["Z"]

        assert Z_new is not None and Z_new.ndim == 2 and Z_new.shape[1] == self.D
        M_new = Z_new.shape[0]

        # Compute the kernel for Z_news
        C   = self.kernel.K(Z, Z)
        Cvv = C + np.diag(1e-6 * np.ones(M))
        Lvv = np.linalg.cholesky(Cvv)

        Cnn = self.kernel.K(Z_new, Z_new)

        # Compute the kernel between the new and valid points
        Cnv = self.kernel.K(Z_new, Z)

        # Predict the psis
        mu_psis_new = np.zeros((self.K, M_new))
        Sig_psis_new = np.zeros((self.K, M_new, M_new))
        for k in range(self.K):
            sys.stdout.write(".")
            sys.stdout.flush()

            psik = data["psi"][:,k]

            # Compute the predictive parameters
            y = solve_triangular(Lvv, psik, lower=True)
            x = solve_triangular(Lvv.T, y, lower=False)
            psik_pred = Cnv.dot(x)

            # Save these into the combined arrays
            mu_psis_new[k] = psik_pred + self.mu[k]

            if full_cov:
                # Sig_pred = Cnn - Cnv.dot(np.linalg.solve(Cvv, Cnv.T))
                Sig_psis_new[k] = Cnn - Cnv.dot(dpotrs(Lvv, Cnv.T, lower=True)[0])

        sys.stdout.write("\n")
        sys.stdout.flush()

        # Convert these to pis
        pis_new = np.array([ln_psi_to_pi(psi) for psi in mu_psis_new])

        if full_output:
            return pis_new, mu_psis_new, Sig_psis_new
        else:
            return pis_new

    def collapsed_predict(self, Z_new):
        raise NotImplementedError

    def resample_model(self, verbose=False):
        self.resample_psi(verbose=verbose)

    def resample_psi(self, verbose=False):
        for data in self.data_list:
            # Compute the cholesky of the covariance matrix
            if data["L"] is None:
                Z = data["Z"]
                C = self.kernel.K(Z) + 1e-6 * np.eye(Z.shape[0])
                L = np.linalg.cholesky(C)
            else:
                L = data["L"]

            # Resample each GP using elliptical slice sampling
            for k in range(self.K):
                if verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()

                # Define a likelihood function for a given value of psi
                def _lkhdk(psik, *args):
                    data["psi"][:,k] = psik
                    return self.log_likelihood(data)

                psik, _ = elliptical_slice(data["psi"][:,k], L, _lkhdk)
                data["psi"][:,k] = psik

            if verbose:
                sys.stdout.write("\n")
                sys.stdout.flush()

    def copy_sample(self):
        # Return psi, omega, and mu
        return self.mu, [data["psi"].copy() for data in self.data_list]

    def set_sample(self, sample):
        mu, psi_omega_list = sample
        self.mu = mu
        assert len(psi_omega_list) == len(self.data_list)
        for psi, data in zip(psi_omega_list, self.data_list):
            data["psi"] = psi


class EmpiricalStickBreakingGPModel(Model):
    """
    Compute the empirical probability given the counts,
    convert the empirical probability into a real valued
    vector that can be modeled with a GP.
    """
    def __init__(self, K, kernel, D=1, alpha=1):
        self.alpha = alpha
        self.K = K
        self.D = D
        self.kernel = kernel

    def add_data(self, Z, X, optimize_hypers=True):

        assert Z.ndim == 2 and Z.shape[1] == self.D
        M = Z.shape[0]
        assert X.shape == (M, self.K)

        # Get the empirical probabilities (offset by 1 to ensure nonzero)
        pi_emp_train = (self.alpha+X).astype(np.float) / \
                       (self.alpha + X).sum(axis=1)[:,None]

        # Convert these to psi's
        self.Z = Z
        self.psi = np.array([pi_to_psi(pi) for pi in pi_emp_train])

        # Compute the mean value of psi
        self.mu = self.psi.mean(axis=0)
        self.psi -= self.mu

        # Create the GP Regression model
        from GPy.models import GPRegression
        self.model = GPRegression(Z, self.psi, self.kernel)

        # Optimize the kernel parameters
        if optimize_hypers:
            self.model.optimize(messages=True)

    def initialize_from_data(self, initialize_to_mle=False):
        "For consistency"
        pass

    def generate(self, keep=True, **kwargs):
        raise NotImplementedError

    def collapsed_predict(self, Z_test):
        psi_pred, psi_pred_var =  self.model.predict(Z_test, full_cov=False)
        psi_pred += self.mu

        pi_pred = np.array([psi_to_pi(psi) for psi in psi_pred])
        return pi_pred, psi_pred, psi_pred_var

    def predict(self, Z_test):
        return self.collapsed_predict(Z_test)

    def predictive_log_likelihood(self, Z_test, X_test):
        pi_pred, _, _ = self.predict(Z_test)

        pll = 0
        pll += gammaln(X_test.sum(axis=1)+1).sum() - gammaln(X_test+1).sum()
        pll += np.nansum(X_test * np.log(pi_pred))

        return pll, pi_pred
