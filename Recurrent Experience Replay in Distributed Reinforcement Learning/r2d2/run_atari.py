"""
From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning
https://openreview.net/pdf?id=r1lyTjAqYX.
"""

from absl import app, flags, logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from deep_rl_zoo.networks.value import R2d2DqnConvNet, RnnDqnNetworkInputs
from deep_rl_zoo.r2d2 import agent
from deel_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop, gym_env, greedy_actors
from deel_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 'Pong', 'Atari name with'
)