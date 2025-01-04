"""
gym environment processing components.
"""

import os
import datetime
import gym.spaces
import gym.wrappers
import numpy as np
import cv2
import logging
import gym
from gym.spaces import Box
from collections import deque
from pathlib import Path

# pylint: disable=import-error
from . import types as types_lib


# A simple list of classic env names.
CLASSIC_ENV_NAMES = [
    'CartPole-v1',
    'LunarLander-v2',
    'MontainCar-v0',
    'Acrobot-v1',
]


def unwrap(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    elif hasattr(env, 'env'):
        return unwrap(env.env)
    elif hasattr(env, 'leg_env'):
        return unwrap(env.leg_env)
    else:
        return env


class NoopReset(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(
        self, env, noop_max=30
    ):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )   # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        return self.env.step(action)


class FireOnReset(gym.Wrapper):
    """
    Take fire action on reset for environments like Breakout.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.render(**kwargs)

        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        return self.env.step(action)


class StickyAction(gym.Wrapper):
    """
    Repeats the last action with epsilon (default 0.25) probability.
    """

    def __init__(self, env, eps=0.25):
        gym.Wrapper.__init__(self, env)
        self.eps = eps
        self.last_action = 0
    
    def step(self, action):
        if np.random.uniform() < self.eps:
            action = self.last_action
        
        self.last_action = action
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)


class LifeLoss(gym.Wrapper):
    """
    Adds boolean key 'loss_life' into the info dict, but only reset on true 
        game over.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_terminated = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_terminated = done

        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            info['loss_life'] = True
        else:
            info['loss_life'] = False
        
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_terminated:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.lives()
        return obs
    

class MaxAndSkip(gym.Wrapper):
    """
    Return only every `skip`-th frame
    """

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )
        self._skip = skip
    
    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grayscale image as done in the Nature paper.
    """

    def __init__(
        self,
        env, 
        width=84,
        height=84,
        grayscale=True,
    ):
        super().__init__(env)

        assert self.observation_space.dtype \
                == np.unit8 and len(self.observation_space.shape) \
                == 3
        
        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.unit8,
        )
    
    def observation(self, obs):
        # pylint: disable=no-member

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA, 
        )
        # pylint: disable=no-member

        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        
        return obs


class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    
    Returns lazy array, which is much more memory efficient.
    See also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shape[:-1] + (shape[-1] * k)),
            dtype=env.observation_space.dtype,
        )
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which 
        can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed
        to the model. You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (
            frames[0].shape[0],
            frames[0].shape[1],
            len(frames)
        )
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]
    
    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]
    
    def frame(self, i):
        return self._force()[..., i]


class ScaleFrame(gym.ObservationWrapper):
    """
    Scale frame by divide 255.
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
    
    def observation(self, obs):
        # Carefull! This undoes the memory optimization, use
        # with smaller replay buffer only.
        return np.array(obs).astype(np.float32) / 255.0


class VisitedRoomInfo(gym.Wrapper):
    """
    Add number of unique visited rooms to the info dictionary.
    For Atari games like MontezumaRevenge and Pitfall.
    """

    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()
    
    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            info['episode_visited_rooms'] = len(self.visited_rooms)
            self.visited_rooms.clear()
        return obs, rew, done, info


class ObscureObservation(gym.ObservationWrapper):
    """
    Make the environment POMDP by obscure the state with probability epsilon.
    this should be used before frame stack.
    """

    def __init__(self, env, epsilon: float = 0.0):
        super().__init__(env)

        if not 0.0 <= epsilon < 1.0:
            raise ValueError(
                f'Expect obscure epsilon should be between [0.0, 1), '
                f'got {epsilon}'
            )
        self._eps = epsilon
    
    def observation(self, obs):
        if self.env.unwrapped.np_random.random() <= self._eps:
            obs = np.zeros_like(
                obs, dtype=self.observation_space.dtype
            )
        return obs


