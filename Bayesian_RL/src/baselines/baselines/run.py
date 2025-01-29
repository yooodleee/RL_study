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


def train(args, extra_args):
    env_type, env_id = get_env_type(args.env)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(
            env,
            osp.join(logger.Logger.CURRENT.dir, "videos"),
            record_video_trigger=lambda x: x % args.save_video_interval == 0,
            video_length=args.save_video_length,
        )
    
    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)
    
    print(
        'Training {} on {}:{} with arguments \n{}'.format(
            args.alg, env_type, env_id, alg_kwargs
        )
    )

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs,
    )

    return model, env


