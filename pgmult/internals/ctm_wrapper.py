

import numpy as np
import os
from os.path import join, basename, isdir
from glob import glob
import subprocess
from textwrap import wrap

from pgmult.utils import mkdir, ln_psi_to_pi
from pgmult.lda import log_likelihood, csr_nonzero


# TODO implement 'seed' init for our models


ctm_url = 'http://www.cs.princeton.edu/~blei/ctm-c/ctm-dist.tgz'
documentfile = 'documents.ldac'
ctmdir = 'deps/ctm-dist'
datadir = os.path.join('data','ctm')
resultsdir = os.path.join('data','ctm','results')
ctm_binary_path = join(ctmdir, 'ctm')
settingsfile = join(ctmdir, 'settings.txt')
logfile = 'ctm-log.txt'

has_ctm_c = os.path.exists(ctm_binary_path)

settings = [
    "em max iter 1000",
    "var max iter 20",
    "cg max iter -1",
    "em convergence 1e-3",
    "var convergence 1e-6",
    "cg convergence 1e-6",
    "lag 1",
    "covariance estimate mle",
]

if not has_ctm_c:
    msg = 'Please download ctm-c from {url} to {ctmdir} and build it. ' \
          '(i.e. the ctm binary should be at {ctm_binary_path})'.format(
        url=ctm_url, ctmdir=ctmdir, ctm_binary_path=ctm_binary_path)
    raise Exception('\n' + '\n'.join(wrap(msg, 82)))

mkdir(os.path.dirname(settingsfile))
with open(settingsfile, 'w') as outfile:
    outfile.writelines('\n'.join(settings))


def dump_ldac_dataset(train_data, datadir):
    mkdir(datadir)

    def pairs(row):
        return ['{}:{}'.format(
                wordid,int(row[0,wordid])) for wordid in row.nonzero()[1]]

    def line(row):
        return '{} {}\n'.format(row.nnz, ' '.join(pairs(row)))

    with open(join(datadir, documentfile), 'w') as outfile:
        outfile.writelines(line(row) for row in train_data if row.nnz > 0)

    return np.array([row.nnz > 0 for row in train_data])


def fit_ldac_ctm(num_topics, datadir, resultsdir):
    if isdir(resultsdir):
        for f in glob(join(resultsdir, '*.dat')):
            os.remove(f)
    mkdir(resultsdir)
    with open(logfile, 'w') as log:
        subprocess.check_call(
            [ctm_binary_path, 'est', join(datadir, documentfile),
                str(num_topics), 'rand', resultsdir, settingsfile],
            stdout=log, stderr=subprocess.STDOUT)


def load_ldac_ctm_results(nonempty, resultsdir):
    K = len(np.atleast_1d(np.loadtxt(join(resultsdir,'final-mu.dat')))) + 1
    results_file_suffixes = dict(
        lmbda=('lambda.dat', (-1,K)),
        nu=('nu.dat', (-1,K)),
        mu=('mu.dat', -1),
        sigma=('cov.dat', (K-1,K-1)),
        log_beta=('log-beta.dat', (K,-1)),
    )

    def load_iter(i):
        return {name:np.loadtxt(join(resultsdir,'%03d-%s' % (i, suffix))).reshape(shape)
                for name, (suffix, shape) in results_file_suffixes.items()}

    def pad_empty(val):
        def pad(param):
            out = np.tile(param.mean(0), (len(nonempty),1))
            out[nonempty] = param
            return out

        return dict(val, nu=pad(val['nu']), lmbda=pad(val['lmbda']))

    iters = sorted(
        [int(fname[:3]) for fname in map(basename,glob(join(resultsdir,'*-mu.dat')))
            if not fname.startswith('final') and fname.endswith('.dat')])
    vals = [pad_empty(load_iter(i)) for i in iters]
    times = np.cumsum(
        np.atleast_2d(np.loadtxt(join(resultsdir,'likelihood.dat'))
                      )[:,3][iters])

    assert len(iters) == len(times) == len(vals)
    return iters, times, vals


def load_mu_sigma(resultsdir):
    mu = np.loadtxt(join(resultsdir,'final-mu.dat')).ravel()
    sigma = np.loadtxt(join(resultsdir,'final-cov.dat')).reshape((mu.shape[0],-1))
    return mu, sigma


def wordprobs(data, val):
    lmbda, beta = val['lmbda'], np.exp(val['log_beta'])
    assert np.allclose(ln_psi_to_pi(lmbda).dot(beta).sum(1), 1.)
    return ln_psi_to_pi(lmbda).dot(beta)[csr_nonzero(data)]


def compute_loglikes(vals, train_data, smoothing):
    return [log_likelihood(train_data, wordprobs(train_data, val))
            for val in vals]


def compute_perplexities(vals, test_data, smoothing):
    return [np.exp(-log_likelihood(test_data, wordprobs(test_data, val))
            / test_data.sum()) for val in vals]


def fit_ctm_em(train, test, T):
    idx = dump_ldac_dataset(train, datadir=datadir)
    fit_ldac_ctm(num_topics=T, datadir=datadir, resultsdir=resultsdir)
    iters, times, vals = load_ldac_ctm_results(idx, resultsdir=resultsdir)
    lls = compute_loglikes(vals, train, smoothing=None)
    perplexities = compute_perplexities(vals, test, smoothing=None)
    plls = compute_loglikes(vals, test, smoothing=None)
    return lls, plls, perplexities, vals, times
