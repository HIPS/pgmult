"""
Test the polya gamma identity:

(e^\psi)^a / (1+e^\psi)^b =
 \int 2^-b e^{kappa*psi -1/2 omega \psi^2} PG(\omega | b, 0) d\omega

using Monte Carlo.
"""


import numpy as np
from scipy.misc import logsumexp
import pypolyagamma as ppg
from pgmult.utils import \
    initialize_polya_gamma_samplers, \
    polya_gamma_density, log_polya_gamma_density

# Set parameters
a = 1
b = 1
psi = -3

def monte_carlo_approx(M=100000):
    ppgs = initialize_polya_gamma_samplers()

    # Compute the left hand side analytically
    loglhs = psi*a - b * np.log1p(np.exp(psi))

    # Compute the right hand side with Monte Carlo
    omegas = np.ones(M)
    ppg.pgdrawvpar(ppgs, b*np.ones(M), np.zeros(M), omegas)
    logrhss = -b * np.log(2) + (a-b/2.)*psi -0.5 * omegas*psi**2
    logrhs = logsumexp(logrhss) - np.log(M)

    print("Monte Carlo")
    print("log LHS: ", loglhs)
    print("log RHS: ", logrhs)

def simps_approx():
    # Compute the left hand side analytically
    loglhs = psi*a - b * np.log1p(np.exp(psi))
    lhs = np.exp(loglhs)

    # Compute the right hand side with quadrature
    from scipy.integrate import simps
    # Lay down a grid of omegas
    # TODO: How should we choose the bins?
    omegas = np.linspace(1e-15, 5, 1000)
    # integrand = lambda om: 2**-b \
    #                        * np.exp((a-b/2.)*psi 0 0.5*om*psi**2) \
    #                        * polya_gamma_density(om, b, 0)
    # y = map(integrand, omegas)
    # rhs = simps(integrand, y)
    logy = -b * np.log(2) + (a-b/2.) * psi -0.5*omegas*psi**2
    logy += log_polya_gamma_density(omegas, b, 0, trunc=21)
    y = np.exp(logy)
    rhs = simps(y, omegas)

    print("Numerical Quadrature")
    print("log LHS: ", loglhs)
    print("log RHS: ", np.log(rhs))

def plot_density():
    omegas = np.linspace(1e-16, 5, 1000)
    logpomega = log_polya_gamma_density(omegas, b, 0, trunc=1000)
    pomega = np.exp(logpomega).real

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    plt.plot(omegas, pomega)

    y = -b * np.log(2) + (a-b/2.) * psi -0.5*omegas*psi**2


    from scipy.integrate import simps
    Z = simps(pomega, omegas)
    print("Z: ", Z)


if __name__ == "__main__":
    monte_carlo_approx(M=100000)
    simps_approx()
    plot_density()


    # debug log p omega calc
    # import ipdb; ipdb.set_trace()
    # logp = log_polya_gamma_density(100, b, 0)
