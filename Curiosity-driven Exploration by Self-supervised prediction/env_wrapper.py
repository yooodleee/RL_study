from __future__ import print_function
import numpy as np
from collections import deque
from PIL import Image
from gym.spaces.box import Box
import gym
import time, sys


class BufferedObsEnv(gym.ObservationWrapper):
    """
    Buffer observations and the buffer, and number of observations stacked.
    skip is the number of steps between buffered observations (min=1).

    n.b. first obs is the oldest, last obs is the newest.
        the buffer is zeroed out on reset.
        *must* call reset() for init!
    """
    def __init__(
        self,
        env=None,
        n=4,
        skip=4,
        shape=(84, 84),
        channel_last=True,
        maxFrames=True,
    ):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        # most recent raw observation (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.maxFrames = maxFrames
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n)
        self.counter = 0    # init and reset should agree on this
        shape = shape + (n,) if channel_last else (n,) + shape
        self.observation_space = Box(0.0, 255.0, shape)
        self.ch_axis = -1 if channel_last else 0
        self.scale = 1.0 / 255
        self.observation_space.high[...] = 1.0
    
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation_space(obs), reward, done, info
    
    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale
    
    def _reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self._convert(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=self.ch_axis)
        return obsNew.astype(np.float32) * self.scale
    
    def _convert(self, obs):
        self.obs_buffer.append(obs)
        if self.maxFrames:
            max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        else:
            max_frame = obs
        intensity_frame = self._rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(
            Image.fromarray(intensity_frame).resize(self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame
    
    def _rgb2y(self, im):
        """
        Converts an RGB image to a Y image (as in YUV).

        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)


class NoNegativeRewardEnv(gym.RewardWrapper):
    """Clip reward in negative direction."""
    def __init__(self, env=None, neg_clip=0.0):
        super(NoNegativeRewardEnv, self).__init__(env)
        self.neg_clip = neg_clip
    
    def _reward(self, reward):
        new_reward = self.neg_clip if reward < self.neg_clip else reward
        return new_reward
    

class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip
    
    def _step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info['steps'] = i + 1
            if done:
                break
        return obs, total_reward, done, info


class MarioEnv(gym.Wrapper):
    def __init__(self, env=None, tilesEnv=False):
        """
        Repeat mario environment without actually restarting fceux everytime.
        This speeds up unrolling by approximately 10 times.
        """
        super(MarioEnv, self).__init__(env)
        self.resetCount = -1
        # reward is distance travelled. So normalize it with total distance
        # https://github.com.ppaquette/gym-super-mario/blob/master/ppaquette_gym_super_mario/lua/super-mario-bros.lua
        # It is only for completion.
        self.maxDistance = 3000.0
        self.tilesEnv = tilesEnv