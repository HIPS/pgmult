
import numpy as np
import time
import re
import os
import operator
import pickle as pickle
import dateutil
from urllib.request import urlopen
from collections import namedtuple

from pybasicbayes.util.text import progprint, progprint_xrange

from pgmult.utils import mkdir
from ctm import get_sparse_repr, split_test_train
from pgmult.lda import StickbreakingDynamicTopicsLDA


#############
#  loading  #
#############

def fetch_sotu():
    baseurl = 'http://stateoftheunion.onetwothree.net/texts/'
    path = 'data/sotu/sotus.pkl'

    def download_text(datestr):
        pagetext = urlopen(baseurl + datestr + '.html').read().replace('\n', ' ')
        paragraphs = re.findall(r'<p>(.*?)</p>', pagetext, re.DOTALL)
        return ' '.join(paragraph.strip() for paragraph in paragraphs)

    if not os.path.isfile(path):
        response = urlopen(baseurl + 'index.html')
        dates = re.findall(r'<li><a href="([0-9]+)\.html">', response.read())

        print('Downloading SOTU data...')
        sotus = {date:download_text(date) for date in progprint(dates)}
        print('...done!')

        mkdir(os.path.dirname(path))
        with open(path, 'w') as outfile:
            pickle.dump(sotus, outfile, protocol=-1)
    else:
        with open(path, 'r') as infile:
            sotus = pickle.load(infile)

    return sotus


def datestrs_to_timestamps(datestrs):
    return [dateutil.parser.parse(datestr).year for datestr in datestrs]


def load_sotu_data(V, sort_data=True):
    sotus = fetch_sotu()
    datestrs, texts = list(zip(*sorted(list(sotus.items()), key=operator.itemgetter(0))))
    return datestrs_to_timestamps(datestrs), get_sparse_repr(texts, V, sort_data)


#############
#  fitting  #
#############

Results = namedtuple(
    'Results', ['loglikes', 'predictive_lls', 'samples', 'timestamps'])


def fit_sbdtm_gibbs(train_data, test_data, timestamps, K, Niter, alpha_theta):
    def evaluate(model):
        ll, pll = \
            model.log_likelihood(), \
            model.log_likelihood(test_data)
        # print '{} '.format(ll),
        return ll, pll

    def sample(model):
        tic = time.time()
        model.resample()
        timestep = time.time() - tic
        return evaluate(model), timestep

    print('Running sbdtm gibbs...')
    model = StickbreakingDynamicTopicsLDA(train_data, timestamps, K, alpha_theta)
    init_val = evaluate(model)
    vals, timesteps = list(zip(*[sample(model) for _ in progprint_xrange(Niter)]))

    lls, plls = list(zip(*((init_val,) + vals)))
    times = np.cumsum((0,) + timesteps)

    return Results(lls, plls, model.copy_sample(), times)


#############
#  running  #
#############

if __name__ == '__main__':
    ## sotu
    # K, V = 25, 2500  # TODO put back
    K, V = 5, 100
    alpha_theta = 1.
    train_frac, test_frac = 0.95, 0.5
    timestamps, (data, words) = load_sotu_data(V)

    ## print setup
    print('K=%d, V=%d' % (K, V))
    print('alpha_theta = %0.3f' % alpha_theta)
    print('train_frac = %0.3f, test_frac = %0.3f' % (train_frac, test_frac))
    print()

    ## split train test
    train_data, test_data = split_test_train(data, train_frac=train_frac, test_frac=test_frac)

    ## fit
    sb_results = fit_sbdtm_gibbs(train_data, test_data, timestamps, K, 100, alpha_theta)

    all_results = {
        'sb': sb_results,
    }

    with open('dtm_results.pkl','w') as outfile:
        pickle.dump(all_results, outfile, protocol=-1)

