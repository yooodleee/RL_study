import sys
import multiprocessing
import os.path as osp
import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.vec_env.vec_frame_stak import VecFramestack
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

from baselines.common.vec_env.cev_normalize import VecNormalize, VecNormalizeRewards


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


try:
    import pybullet_envs    # more info: https://github.com/bulletphysics/bullet3/blob/master/README.md
except ImportError:
    pybullet_envs = None


try:
    import roboschool
except ImportError:
    roboschool = None


_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes

    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)


# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


