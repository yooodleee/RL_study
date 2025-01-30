"""
Helpers for scripts like run_atari.py.
"""

import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenDictWrapper
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subprob_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common import retro_wrappers

