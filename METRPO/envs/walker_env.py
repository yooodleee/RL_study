import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env


class Walker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (
            height > 0.8 and height < 2.0 and
            ang > -1.0 and ang < 1.0
        )
        ob = self._get_obs()
        return ob, reward, done, {}
    
    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
    
    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv),
        )
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
    
    @property
    def n_goals(self):
        """
        return: goal dimensions
        """
        return 2
    
    @property
    def n_states(self):
        """
        return: state dimensions
        """
        return 4
    
    def cost_np(
            self,
            x,
            u,
            x_next,
            ctrl_cost_coeff=2):
        
        assert np.amax(np.abs(u)) <= 1.0
        return np.mean(np.linalg.norm(x[:, -2:] - get_fingertips(x), axis=1) \
                       + ctrl_cost_coeff * 0.5 * np.sum(np.square(u), axis=1))
    
    def cost_tf(
            self,
            x,
            u,
            x_next,
            ctrl_cost_coeff=2):
        
        return tf.reduce_mean(tf.norm(x[:, -2:] - get_fingertips_tf(x), axis=1) \
                              + ctrl_cost_coeff * 0.5 * tf.reduce_sum(tf.square(u), axis=1))
    
    