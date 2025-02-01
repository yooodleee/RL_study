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
    
    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))
        
        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(
                n_envs=n_envs,
                max_path_length=self.algo.max_path_length,
            )
        else:
            envs = [
                pickle.loads(pickle.dumps(self.algo.env))
                for _ in range(n_envs)
            ]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length,
            )
        self.env_spec = self.algo.env.spec
    
    def shutdown_worker(self):
        self.vec_env.terminate()
    
    