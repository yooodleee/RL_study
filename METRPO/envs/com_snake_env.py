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
    
    def get_current_obs(self):
        qpos = np.squeeze(self.model.data.qpos)
        qvel = np.squeeze(self.model.data.qvel)
        return np.concatenate([
            self.get_body_com("torso")[:2],
            qpos[2:],
            self.get_body_comvel("torso")[:2],
            qvel[2:],
        ]).reshape(-1)
    
    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]
    
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False

        return Step(next_obs, reward, done)
    
    @overrides
    def log_diagnostics(self, paths, prefix=''):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)
    
    def cost_np(
            self,
            x,
            u,
            x_next):
        
        assert int(x.shape[1] / 2) == idx
        assert np.amax(np.abs(u)) <= 1.0
        return -np.mean(
            x_next[:, idx] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1)
        )
    
    def cost_tf(
            self,
            x,
            u,
            x_next):
        
        return -tf.reduce_mean(
            x_next[:, idx] - self.ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1)
        )
    
    def cost_np_vec(
            self,
            x,
            u,
            x_next):
        
        assert int(x.shape[1] / 2) == idx
        assert np.amax(np.abs(u)) <= 1.0
        return -(x_next[:, idx] - self.ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1))