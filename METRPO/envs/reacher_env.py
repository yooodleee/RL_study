import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import tensorflow as tf


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
    
    # def _step(self, a):
    #     x = self._get_bos()[None]
    #     assert np.allclose(self.get_body_com("fingertip")[:2], get_fingertips(x)), \
    #             str(self.get_body_com("fingertip")) + " " + str(get_fingertips(x))
    #     vec = self.get_body_com("fingertip") - self.get_body_com("target")
    #     reward_dist = -np.linalg.norm(vec[:2])
    #     reward_ctrl = -np.square(a).sum() * 0.01
    #     reward = reward_dist + reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _step(self, a):
        a = np.reshape(np.clip(a, -1, 1), -1)
        obs = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        obs_next = self._get_obs()
        reward = -self.cost_np(obs[None], a[None], obs_next[None])
        done = False

        return obs_next, reward, done, dict(ctrl_cost=np.sum(np.square(a)), comment='this_is_local_step')
    
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
    
    