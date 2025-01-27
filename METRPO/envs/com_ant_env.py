from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math
import tensorflow as tf


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, *args, **kwargs):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
    
    def get_current_obs(self):
        return np.concatenate([
            self.get_body_com("torso").flat,
            self.model.data.qpos.flat[3:],
            self.get_body_comvel("torso").flat,
            self.model.data.qvel.flat[3:],
            # np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            # self.get_body_xmat("torso").flat,
            # self.get_body_com("torso"),
        ]).reshape(-1)
    
    def step(self, action):
        self.forward_dynamics(action)
        com = self.get_body_com("torso")
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        # contact_cst = 0.5 * 1e-3 * np.sum(
        #       np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and com[2] >= 0.2 and com[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()

        return Step(ob, float(reward), done)
    
    