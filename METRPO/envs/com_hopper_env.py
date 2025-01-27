import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


# states: [
# 0: z-coord,
# 1: x-coord (forward distance),
# 2: forward pitch along y-axis,
# 6: z-vel (up = +),
# 7: xvel (forward = +)


class HooperEnv(MujocoEnv, Serializable):

    FILE = 'hopper.xml'

    @autoargs.arg(
        'alive_coeff',
        type=float,
        help='reward coefficient for being alive',
    )

    @autoargs.arg(
        'ctrl_cost_coeff',
        type=float,
        help='cost coefficient for controls',
    )

    def __init__(
            self,
            alive_coeff=1,
            ctrl_cost_coeff=0.01,
            *args,
            **kwargs):
        
        self.alive_coeff = alive_coeff
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(HooperEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
    
    