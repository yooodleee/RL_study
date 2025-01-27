import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import tensorflow as tf


BIG = 1e6
idx = 7


class SnakeEnv(MujocoEnv, Serializable):
    FILE = 'snake.xml'
    ORI_IND = 2

    @autoargs.arg(
        'ctrl_cost_coeff',
        type=float,
        help='cost coefficient for controls',
    )

    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            ego_obs=False,
            sparse_rew=False,
            *args,
            **kwargs):
        
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.ego_obs = ego_obs
        self.sparse_rew = sparse_rew
        super(SnakeEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
    
    