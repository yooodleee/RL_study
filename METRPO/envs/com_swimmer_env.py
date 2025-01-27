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


def get_original_representation(states):
    """
    The first two are com and the rest are angles.
    """
    if states is None:
        return None
    assert len(states) == 10
    out = np.array(states)
    out[:2] -= 2 / 3 * get_xy_coordinate(states[2])
    out[:2] -= 1 / 2 * get_xy_coordinate(np.pi + states[2] + states[3])
    out[:2] -= 1 / 6 * get_xy_coordinate(np.pi + states[2] + states[3] + states[4])
    out[5:7] -= 2 / 3 * get_dxy_by_dt(states[2], states[7])
    out[5:7] -= 1 / 2 * get_dxy_by_dt(
        np.pi + states[2] + states[3], states[7] + states[8]
    )
    out[5:7] -= 1 / 6 * get_dxy_by_dt(
        np.pi + states[2] + states[3] + states[4], states[7] + states[8] + states[9]
    )
    
    return out


class SwimmerEnv(MujocoEnv, Serializable):

    FILE = 'swimmer.xml'
    ORI_IND = 2

    @autoargs.arg(
        'ctrl_cost_coeff',
        type=float,
        help='cost coefficient for controls',
    )

    def __init__(
            self,
            ctrl_cost_coeff=1e-2,
            *args,
            **kwargs):
        
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(SwimmerEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
    
    def get_original_states(self):
        qpos = np.squeeze(self.model.data.qpos)
        qvel = np.squeeze(self.model.data.qvel)
        return np.concatenate([qpos, qvel])
    
    def get_current_obs(self):
        qpos = np.squeeze(self.model.data.qpos)
        qvel = np.squeeze(self.model.data.qvel)
        return np.concatenate([
            self.get_body_com("torso")[:2],
            qpos[2:5],
            self.get_body_comvel("torso")[:2],
            qvel[2:5],
        ]).reshape(-1)
    
    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]
    
    