from . import VecEnvWrapper
import numpy as np
from gym import spaces


class VecFrameStack(VecEnvWrapper):

    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space    # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)

        self.stackedobs = np.zeros(
            (venv.num_envs,) + low.shape, low.dtype
        )
        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)
    
    