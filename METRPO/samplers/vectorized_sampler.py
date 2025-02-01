import pickle

import tensorflow as tf
from samplers.base import BaseSampler
from envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs
    
    def set_init_sampler(self, init_sampler):
        self.vec_env.vec_env.set_init_sampler(init_sampler)
    
    