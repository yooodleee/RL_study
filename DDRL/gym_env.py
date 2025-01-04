"""
gym environment processing components.
"""

import os
import datetime
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


