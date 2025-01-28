import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import tensorflow as tf


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
    
    