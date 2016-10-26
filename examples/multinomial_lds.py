
import numpy as np
import matplotlib.pyplot as plt

from pgmult.lds import MultinomialLDS
from pybasicbayes.distributions import Gaussian, AutoRegression

np.seterr(invalid="warn")
np.random.seed(0)

#########################
#  set some parameters  #
#########################

T = 200
D = 2
mu_init = np.array([0.,1.])
sigma_init = 0.01*np.eye(D)

A = 0.999*np.array([[np.cos(np.pi/24), -np.sin(np.pi/24)],
                   [np.sin(np.pi/24),  np.cos(np.pi/24)]])
sigma_states = 0.0001*np.eye(D)

K = 4
# C = np.hstack((np.ones((K-1, 1)), np.zeros((K-1, D-1))))
C = np.random.randn(K-1, D)
sigma_obs = 0.01 * np.eye(K)

###################
#  generate data  #
###################

truemodel = MultinomialLDS(K, D,
    init_dynamics_distn=Gaussian(mu=mu_init,sigma=sigma_init),
    dynamics_distn=AutoRegression(A=A,sigma=sigma_states),
    C=C
    )

data = truemodel.generate(T=T)


###################
#    inference    #
###################
testmodel = MultinomialLDS(K, D,
    init_dynamics_distn=Gaussian(mu_0=mu_init, sigma_0=sigma_init, kappa_0=1.0, nu_0=3.0),
    dynamics_distn=AutoRegression(nu_0=D+1,S_0=np.eye(D),M_0=np.zeros((D,D)),K_0=np.eye(D)),
    sigma_C=1
    )

testmodel.add_data(data["x"])
testdata = testmodel.data_list[0]

N_samples = 100
samples = []
lls = []
pis = []
psis = []
zs = []
for smpl in range(N_samples):
    print("Iteration ", smpl)
    testmodel.resample_model()

    samples.append(testmodel.copy_sample())
    lls.append(testmodel.log_likelihood())
    pis.append(testmodel.pi(testdata))
    psis.append(testmodel.psi(testdata))
    zs.append(testdata["states"].stateseq)

lls = np.array(lls)
pis = np.array(pis)

psis = np.array(psis)
psi_mean = psis[N_samples//2:,...].mean(0)
psi_std = psis[N_samples//2:,...].std(0)

zs = np.array(zs)
z_mean = zs[N_samples//2:,...].mean(0)
z_std = zs[N_samples//2:,...].std(0)


# Plot the true and inferred states
plt.figure()
ax1 = plt.subplot(411)
plt.errorbar(np.arange(T), z_mean[:,0], color="r", yerr=z_std[:,0])
plt.errorbar(np.arange(T), z_mean[:,1], ls="--", color="r", yerr=z_std[:,1])
plt.plot(data["states"].stateseq[:,0], '-b', lw=2)
plt.plot(data["states"].stateseq[:,1], '--b', lw=2)
ax1.set_title("True and inferred latent states")

ax2 = plt.subplot(412)
plt.imshow(data["x"].T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
ax2.set_title("Observed counts")

ax3 = plt.subplot(413)
plt.imshow(truemodel.pi(data).T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
ax3.set_title("True probabilities")

ax4 = plt.subplot(414)
plt.imshow(pis[N_samples//2:,...].mean(0).T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
ax4.set_title("Mean inferred probabilities")

# Plot the log likelihood
plt.figure()
plt.plot(lls)
plt.xlabel('Iteration')
plt.ylabel("Log likelihood")

plt.show()

