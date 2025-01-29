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


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args.env)

    print(env_id)
    # extract the agc_env_name
    noskip_idx = env_id.find("NoFrameskip")
    env_name = env_id[:noskip_idx].lower()
    print("Env Name for Masking: ", env_name)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(
                env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale
            )
            env = VecFramestack(env, frame_stack_size)

    else:
        print("preconfig")
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
        )

        print("post config")
        config.gpu_options.allow_growth = True
        get_session(config=config)
        print("got session")
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale)
        print("made env")
    if args.custom_reward != '':
        from baselines.common.vec_env import VecEnv, VecEnvWrapper
        import baselines.common.custom_reward_wrapper as W
        assert isinstance(env, VecEnv) or isinstance(env, VecEnvWrapper)

        custom_reward_kwargs = eval(args.custom_reward_kwargs)

        if args.custom_reward == 'live_long':
            env = W.VecLiveLongReward(env, **custom_reward_kwargs)
        elif args.custom_reward == 'random_tf':
            env = W.VecTFRandomReward(env, **custom_reward_kwargs)
        elif args.custom_reward == 'preference':
            env = W.VecTFPreferenceReward(env, **custom_reward_kwargs)
        elif args.custom_reward == 'rl_irl':
            if args.custom_reward_path == '':
                assert False, 'no path for reward model'
            else:
                if args.custom_reward_lambda == '':
                    assert False, 'no combination parameter lambda'
                else:
                    env = W.VecRLplusIRLAtariReward(env, args.custom_reward_path, args.custom_reward_lambda)
        elif args.custom_reward == 'pytorch':
            if args.custom_reward_path == '':
                assert False, 'no path for reward model'
            else:
                if env_type == "atari":
                    env = W.VecPyTorchAtariReward(env, args.custom_reward_path, env_name)
                elif env_type == "mujoco":
                    env = W.VecPyTorchMujocoReward(env, args.custom_reward_path, env_name)
        elif args.custom_reward == "mcmc_mean":
            if args.custom_reward_path == '' or args.mcmc_chain_path == '':
                assert False, 'no path for reward model and/or chain_path'
            else:
                env = W.VecMCMCMeanAtariReward(env, args.custom_reward_path, args.mcmc_chain_path, args.embedding_dim, env_name)
        elif args.custom_reward == "mcmc_map":
            if args.custom_reward_path == '':
                assert False, 'no path for reward model and/or chain_path'
            else:
                env = W.VecMCMCMAPAtarireward(env, args.custom_reward_path, args.embedding_dim, env_name)
        else:
            assert False, 'no such wrapper exist'
    
    if env_type == 'mujoco':
        print("normalized environment")
        env = VecNormalize(env)
    # if env_type == 'atari':
    #       input("Normalizing for Atari game: okay? [Enter]")
    #       # normalize rewards but not observations for atari
    #       env = VecNormalizeRewards(env)

    return env


