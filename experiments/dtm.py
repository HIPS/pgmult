from __future__ import division
import re
import os
import sys
import operator
from urllib2 import urlopen
import cPickle as pickle

from pgmult.internals.utils import mkdir
from ctm import get_sparse_repr


##########
#  util  #
##########

def progprint(itr):
    for item in itr:
        sys.stdout.write('.')
        sys.stdout.flush()
        yield item


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

        print 'Downloading SOTU data...'
        sotus = {date:download_text(date) for date in dates}
        print '...done!'

        with open(path, 'w') as outfile:
            pickle.dump(sotus, outfile, protocol=-1)
    else:
        mkdir(os.path.dirname(path))
        with open(path, 'r') as infile:
            sotus = pickle.load(infile)

    return sotus


def load_sotu_data(V, sort_data=True):
    sotus = fetch_sotu()
    datestrs, texts = zip(*sorted(sotus.items(), key=operator.itemgetter(0)))
    return datestrs, get_sparse_repr(texts, V, sort_data)
