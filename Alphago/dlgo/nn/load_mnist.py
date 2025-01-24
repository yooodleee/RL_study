import six.moves.cPickle as pickle
import gzip
import numpy as np


def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


