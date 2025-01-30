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


class EzPickle(object):
    """
    Objects that are pickled and unpickled via their constructor
        arguments.

    Example usage:

        class Dog(Animal, EzPickle):
            def __init__(self, furcolor, tailkind="bushy"):
                Animal.__init__()
                EzPickle.__init__(furcolor, tailkind)
    
    When this object is unpickled, a new Dog will be constructed by passing
        the provided furcolor and tailkind into the constructor. However,
        philosophers are still not sure whether it is still the same dog.

    This is generally needed only for environments which wrap C/C++ code, 
        such as MuJoCo and Atari.
    """
    def __init__(self, *args, **kwargs):
        self._ezpickle_args = args
        self._ezpickle_kwargs = kwargs
    
    def __getstate__(self):
        return {
            "_ezpickle_args": self._ezpickle_args,
            "_ezpickle_kwargs": self._ezpickle_kwargs,
        }
    
    def __setstate__(self, d):
        out = type(self)(*d["_ezpickle_args"], **d["_ezpickle_kwargs"])
        self.__dict__.update(out.__dict__)


def set_global_seeds(i):
    try:
        from mpi4py import MPI
    except ImportError:
        rank = 0
    
    myseed = i + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        
        tf.compat.v1.set_random_seed(myseed)
    except ImportError:
        pass

    np.random.seed(myseed)
    random.seed(myseed)


def pretty_eta(seconds_left):
    """
    Print the number of seconds in human readable format.

    Examples:
    2 days
    2 hours and 37 minutes
    less than a minute

    Parameters
    -----------
    seconds_left: int
        Number of seconds to be converted to the ETA

    Returns
    -----------
    eta: str
        String representing the pretty ETA.

    """
    minutes_lefts = seconds_left // 60
    seconds_left %= 60
    hours_left = minutes_lefts // 60
    minutes_lefts %= 60
    days_left = hours_left // 24
    hours_left %= 24


    def helper(cnt, name):
        return "{} {}{}".format(str(cnt), name, ('s' if cnt > 1 else ''))
    

    if days_left > 0:
        msg = helper(days_left, 'day')
        if hours_left > 0:
            msg += ' and ' + helper(hours_left, 'hour')
        return msg
    
    if hours_left > 0:
        msg = helper(hours_left, 'hour')
        if minutes_lefts > 0:
            msg += ' and ' + helper(minutes_lefts, 'minute')
        return msg
    
    if minutes_lefts > 0:
        return helper(minutes_lefts, 'minute')
    
    
    return 'less than a minute'


