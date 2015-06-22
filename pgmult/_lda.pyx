# distutils: extra_compile_args = -O3 -w
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf, fflush, stdout

TOPIC = np.uint32
ctypedef np.uint32_t TOPIC_t

WORD = np.uint32
ctypedef np.uint32_t WORD_t

COUNT = np.uint32
ctypedef np.uint32_t COUNT_t


cdef inline TOPIC_t sample_discrete(double[::1] probs, double tot):
    cdef TOPIC_t i
    tot *= (<double> rand()) / RAND_MAX
    for i in range(probs.shape[0]):
        tot -= probs[i]
        if tot < 0.:
            return i
    return i

cdef class CollapsedCounts(object):
    # python-accessible attributes
    property word_topic_counts:
        def __get__(self):
            return np.asarray(self.word_topic_c)

    property doc_topic_counts:
        def __get__(self):
            return np.asarray(self.document_topic_c)

    property z:
        def __get__(self):
            return self.reconstruct_z()

    # internal parameters and counts
    cdef double alpha_theta, alpha_beta
    cdef int T, V, D

    cdef WORD_t[::1] words
    cdef TOPIC_t[::1] labels
    cdef int[::1] doc_starts
    cdef int[::1] doc_ends

    cdef COUNT_t[::1] topic_c,
    cdef COUNT_t[:,::1] document_topic_c,
    cdef COUNT_t[:,::1] word_topic_c

    cdef object csr_data

    # pre-allocated buffer because cython can't do dynamic stack arrays
    cdef double[::1] topic_scores_buf

    @cython.wraparound(True)
    def __init__(
            self, double alpha_theta, double alpha_beta, int T,
            np.ndarray z, np.ndarray doc_topic_counts, np.ndarray word_topic_counts,
            object csr_data):
        self.alpha_theta = alpha_theta
        self.alpha_beta = alpha_beta
        self.T = T
        self.D, self.V = csr_data.shape
        self.csr_data = csr_data

        self.topic_scores_buf = np.empty(T,dtype=np.double)

        # replicate out a words vector that matches the given word counts for
        # each document, concatenating across documents
        self.words = csr_data.indices.repeat(csr_data.data).astype(WORD)

        # make a corresponding labels vector consistent with the given z for
        # each document, concatenating across documents
        self.labels = np.concatenate(
            [np.arange(T).repeat(doc) for doc in z]).astype(TOPIC)

        # set up document start and stop indices for those concatenated arrays
        indices = np.concatenate(
            ((0,), np.cumsum(np.asarray(csr_data.sum(1)).ravel()))).astype(np.int32)
        self.doc_starts, self.doc_ends = indices[:-1], indices[1:]

        # count up the z's (or labels) into cached summary statistics
        self.document_topic_c = doc_topic_counts.astype(COUNT)
        self.word_topic_c = word_topic_counts.astype(COUNT)
        self.topic_c = np.sum(self.word_topic_c,axis=0).astype(COUNT)

    def resample(self, int niter):
        cdef int itr, doc, i
        for itr in range(niter):
            for doc in range(self.D):
                for i in range(self.doc_starts[doc],self.doc_ends[doc]):
                    self.count(self.labels[i],self.words[i],doc,-1)
                    self.labels[i] = self.sample_topic(self.words[i],doc)
                    self.count(self.labels[i],self.words[i],doc,1)

    cdef inline TOPIC_t sample_topic(self, WORD_t word, int doc_id):
        cdef TOPIC_t t
        cdef double score, tot = 0.
        for t in range(self.T):
            score = self.score(t,word,doc_id)
            self.topic_scores_buf[t] = score
            tot += score
        return sample_discrete(self.topic_scores_buf,tot)

    cdef inline void count(self, TOPIC_t topic, WORD_t word, int doc_id, int inc):
        self.topic_c[topic] += inc
        self.word_topic_c[word,topic] += inc
        self.document_topic_c[doc_id,topic] += inc

    cdef inline double score(self, TOPIC_t topic, WORD_t word, int doc_id):
        return (self.alpha_theta + self.document_topic_c[doc_id,topic]) \
                * (self.alpha_beta + self.word_topic_c[word,topic]) \
                  / (self.alpha_beta*self.V + self.topic_c[topic])

    @cython.wraparound(True)
    cdef reconstruct_z(self):
        cdef np.uint32_t[:,::1] z = np.zeros((self.csr_data.nnz, self.T), dtype=np.uint32)
        cdef np.uint32_t[::1] lens = self.csr_data.data
        cdef np.uint32_t[::1] starts = np.concatenate(((0,), np.cumsum(lens[:-1]))).astype(np.uint32)
        cdef int i, j

        for i in range(starts.shape[0]):
            for j in range(starts[i], starts[i] + lens[i]):
                z[i, self.labels[j]] += 1

        return np.asarray(z)

