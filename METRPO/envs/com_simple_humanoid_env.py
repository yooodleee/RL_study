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
    
    def step(self, action):
        self.forward_dynamics(action)

        next_obs = self.get_current_obs()

        alive_bonus = self.alive_bonus
        data = self.model.data

        # comvel = self.get_body_comvel("torso")
        idx = self.model.geom_names.index("head")
        head_h = self.model.data.geom_xpos[idx][-1]

        # lin_vel_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 1e-2 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        # impact_cst = .5 * self.impact_cost_coeff * np.sum(np.square(np.clip(data.cfrc_ext, -1, 1)))
        # vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(np.square(comvel[1:]))
        # reward = lin_vel_reward + alive_bonus - ctrl_cost - impact_cost - vel_deviation_cost
        # done = data.qpos[2] < 0.8 or data.qpos[2] > 2.0
        reward = -(head_h - 1.5) ** 2 - ctrl_cost
        done = False

        return Step(next_obs, reward, done)
    
    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))
    
    def cost_np(
            self,
            x,
            u,
            x_next):
        
        assert np.amax(np.abs(u)) <= 1.0
        return np.mean(self.cost_np_vec(x, u, x_next))
    
    def cost_tf(
            self,
            x,
            u,
            x_next):
        
        ctrl_cost = 1e-2 * self.ctrl_cost_coeff * tf.reduce_sum(np.square(u), axis=1)
        return tf.reduce_mean((x_next[:, -1] - 1.5) ** 2 + ctrl_cost)
    
    def cost_np_vec(
            self,
            x,
            u,
            x_next):
        
        assert np.amax(np.abs(u)) <= 1.0
        head_h = x_next[:, -1]
        ctrl_cost = 1e-2 * self.ctrl_cost_coeff * np.sum(np.square(u), axis=1)
        return (head_h - 1.5) ** 2 + ctrl_cost