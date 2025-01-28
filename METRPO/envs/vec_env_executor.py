import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils


class VecEnvExecutor(object):

    def __init__(
            self,
            envs,
            max_path_length):
        
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length
    
    