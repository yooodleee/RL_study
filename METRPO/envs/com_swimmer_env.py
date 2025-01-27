from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import tensorflow as tf


def get_xy_coordinate(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def get_dxy_by_dt(theta, theta_dot):
    return np.array([-np.sin(theta) * theta_dot, np.cos(theta) ( theta_dot)])


