from samplers.base import BaseSampler
from rllab.sampler import parallel_sampler
from rllab.sampler.stateful_pool import singleton_pool
import tensorflow as tf


def worker_init_tf(G):
    G.sess = tf.compat.v1.Session()
    G.sess.__enter__()


