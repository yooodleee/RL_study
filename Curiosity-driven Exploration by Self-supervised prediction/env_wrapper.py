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
    def __init__()