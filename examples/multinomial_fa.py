import numpy as np
import matplotlib.pyplot as plt

from pgmult.fa import MultinomialFA

np.seterr(invalid="warn")
np.random.seed(0)

#########################
#  set some parameters  #
#########################
M = 200
D = 1
K = 4
N = 10
C = np.random.randn(K-1, D)
sigma_obs = 0.01 * np.eye(K)

###################
#  generate data  #
###################
truemodel = MultinomialFA(K, D, C=C)
truedata = truemodel.generate(M=M, N=N, full_output=True)
x, z = truedata["x"], truedata["z"]


###################
#    inference    #
###################
testmodel = MultinomialFA(K, D, sigma_C=1)
testmodel.add_data(x)
testdata = testmodel.data_list[0]

N_samples = 100
samples = []
lls = []
pis = []
psis = []
zs = []
for smpl in xrange(N_samples):
    print "Iteration ", smpl
    testmodel.resample_model()

    samples.append(testmodel.copy_sample())
    lls.append(testmodel.log_likelihood())
    pis.append(testmodel.pi(testdata))
    psis.append(testmodel.psi(testdata))
    zs.append(testdata["z"])

lls = np.array(lls)
pis = np.array(pis)

psis = np.array(psis)
psi_mean = psis[N_samples//2:,...].mean(0)
psi_std = psis[N_samples//2:,...].std(0)

zs = np.array(zs)
z_mean = zs[N_samples//2:,...].mean(0)
z_std = zs[N_samples//2:,...].std(0)


###################
#    plotting     #
###################
plt.figure()
ax1 = plt.subplot(411)
plt.errorbar(np.arange(M), z_mean[:,0], color="r", yerr=z_std[:,0])
plt.plot(z[:,0], '-b', lw=2)
plt.plot(np.zeros(M), ':k', lw=0.5)
ax1.set_title("True and inferred latent states")

ax2 = plt.subplot(412)
plt.imshow(x.T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
ax2.set_title("Observed counts")

ax3 = plt.subplot(413)
plt.imshow(truemodel.pi(truedata).T, interpolation="none", vmin=0, vmax=1, cmap="Blues")
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

# C_true = truemodel.emission_distn.C
# C_test = testmodel.emission_distn.C
# print "C_true: ", C_true
# print "C_test: ", C_test
