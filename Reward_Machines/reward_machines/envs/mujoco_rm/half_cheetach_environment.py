"""
Code add event detectors to the Ant3 Env.
"""

import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from reward_machines.rm_environment import RewardMachineEnv


class MyHalfCheetahEnv(gym.Wrapper):

    def __init__(self):
        """
        Note that the current position is key for this task.
        """
        super.__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))
    
    