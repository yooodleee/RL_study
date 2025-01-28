import numpy as np
import tensorflow as tf
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
from rllab.core.serializable import Serializable


class PointMassEnv(Env, Serializable):
    
    def __init__(self):
        Serializable.quick_init(self, locals())
        self.qpos = None
        self.qvel = None
        self.mass = 0.1
        self.dt = 0.01
        self.frame_skip = 5
        self.boundary = np.array([-10, 10])
        self.vel_bounds = [-np.inf, np.inf]
        """
        In 1 frame forward,
            qpos' = qpos + qvel * dt
            qvel' = qvel + u/m * dt
        """
        eig_vec = np.array([[0.7, -0.6], [-0.3, -0.1]])
        self.A = np.identity(2) # eig_vec @ np.diag([1.0, 0.8]) @ np.linalg.inv(eig_vec)
        self.B = np.array([[0.2, -0.04], [.3, .9]])
        self.c = np.array([0.0, 0.0])
        self.goal = None
        self.init_mean = np.zeros(2)
        self.init_std = 0.1
        self.ctrl_cost_coeff = 0.01
    
    