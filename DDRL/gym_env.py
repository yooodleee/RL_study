"""
gym environment processing components.
"""

import os
import datetime
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
    

