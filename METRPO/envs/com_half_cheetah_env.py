import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(MujocoEnv, Serializable):

    FILE = 'half_cheetah.xml'

    def __init__(self, *args, **kwargs):
        super(HalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.ctrl_cost_coeff = 1e-1
    
    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso")[[0, 2]],
            self.model.data.qpos.flatten()[2:],
            self.get_body_comvel("torso")[[0, 2]],
            self.model.data.qvel.flatten()[2:],
        ])
    
    