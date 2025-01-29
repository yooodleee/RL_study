import gym
import numpy as np
import os
import pickle
import random
import tempfile
import zipfile


def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)


def unpack(seq, sizes):
    """
    Unpack 'seq' into a sequnece of lists, with lengths specified by 'sizes'.
    None = just one bare element, not a list.

    Example:
        unpack([1, 2, 3, 4, 5, 6], [3, None, 2]) -> ([1, 2, 3], 4, [5, 6])

    """
    seq = list(seq)
    it = iter(seq)
    assert sum(1 if s is None else s for s in sizes) == len(seq), "Trying to unpack %s into %s" % (seq, sizes)
    for size in sizes:
        if size is None:
            yield it.__next__()
        else:
            li = []
            for _ in range(size):
                li.append(it.__next__())
            yield li


