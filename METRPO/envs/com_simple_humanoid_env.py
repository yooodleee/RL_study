from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import autoargs


class SimpleHumanoidEnv(MujocoEnv, Serializable):

    FILE = 'simple_humanoid.xml'

    @autoargs.arg(
        'vel_deviation_cost_coeff',
        type=float,
        help='cost coefficient for velocity deviation',
    )
    @autoargs.arg(
        'alive_bonus',
        type=float,
        help='bonus reward for being alive',
    )
    @autoargs.arg(
        'ctrl_cost_coeff',
        type=float,
        help='cost coefficient for control inputs',
    )
    @autoargs.arg(
        'impact_cost_coeff',
        type=float,
        help='cost coefficient for impact',
    )

    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            *args,
            **kwargs):
        
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        super(SimpleHumanoidEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
    
    def get_current_obs(self):
        data = self.model.data
        idx = self.model.geom_names.index("head")
        head_ops = self.data.geom_xpos[idx]
        return np.concatenate([
            data.qpos.flat[3:],
            data.qvel.flat,
            # np.clip(data.cfrc_ext, -1, 1).flat,
            # self.get_body_com("torso").flat,
            head_ops,
        ])
    
    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]
    
    