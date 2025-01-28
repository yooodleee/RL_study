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
    
    def reset(self, init_state=None):
        if init_state is None:
            self.qpos = self.init_mean + np.random.randn(2) * self.init_std
            self.qvel = self.init_mean + np.random.randn(2) * self.init_std
            # random goal in [5, 10] and [-5, -10]
            self.goal = np.random.uniform(-self.boundary, self.boundary) \
                        * ((np.random.uniform(size=2) > 0.5).astype(np.float32) * 2 - 1.)
        else:
            assert len(init_state) == 6
            self.qpos = init_state[:2]
            self.qvel = init_state[2:4]
            self.goal = init_state[4:]
        return self.get_obs()
    
    def step(self, action):
        assert np.all(self.qpos) and np.all(self.qvel) and np.all(self.goal), \
            "call env.reset before step"
        # Clipping action
        action = np.clip(action, *self.action_space.bounds)
        action = np.reshape(action, -1)
        qpos = self.qpos
        qvel = self.qvel
        for i in range(self.frame_skip):
            qpos = np.clip(qpos + qvel * self.dt, *self.boundary)
            # qvel = np.clip(qvel + (action / self.mass) * self.dt, *self.vel_bounds)
            qvel = np.clip(self.A@qvel + self.B@action + self.c, *self.vel_bounds)
        self.qpos = qpos
        self.qvel = qvel
        return Step(
            observation=self.get_obs(),
            reward=self.get_reward(action),
            done=False,
        )
    
    def get_obs(self):
        return np.concatenate(
            [self.qpos, self.qvel, self.goal], axis=0
        )
    
    def get_reward(self, action):
        """
        Distance from goal and action cost.
        """
        cost = np.linalg.norm(self.goal - self.qpos) \
                + self.ctrl_cost_coeff \
                * np.mean(np.square(action), axis=0)
        return -cost
    
    