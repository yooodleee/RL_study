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
import types as types_lib


# A simple list of classic env names.
CLASSIC_ENV_NAMES = [
    'CartPole-v1',
    'LunarLander-v2',
    'MontainCar-v0',
    'Acrobot-v1',
]