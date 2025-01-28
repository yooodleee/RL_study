import numpy as np
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import tensorflow as tf


ctrl_cost_coeff = 0.01
goal = np.array([8., 5.])


class Point2DEnv(Env, Serializable):

    def __init__(self):
        Serializable.quick_init(self, locals())
        self.state = None
        """
        x_next = Ax + Bu + c + D*noise
        """
        self.transition = {
            'A': np.array([[1., 0.03], [0., 1.]]),
            'B': np.array([[1., 0.], [0., 1.]]),
            'c': np.array([0., 0.]),
        }
        self.goal = goal
        self.init_mean = np.zeros(2)
        self.init_std = 0.1
        self.ctrl_cost_coeff = ctrl_cost_coeff
    
    