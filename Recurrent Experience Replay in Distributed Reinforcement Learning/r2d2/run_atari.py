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
    'environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Sequest.'
)
flags.DEFINE_integer(
    'environment_height', 84, 'Environment frame screen height.'
)
flags.DEFINE_integer(
    'environment_width', 84, 'Environment frame screen width.'
)
flags.DEFINE_integer(
    'environment_frame_skip', 4, 'Number of frames to skip.'
)
flags.DEFINE_integer(
    'environment_frame_stack', 1, 'Number of frames to stack.'
)
flags.DEFINE_bool(
    'compress_state', True, 'Compress state images when store in experience replay.'
)
flags.DEFINE_integer(
    'num_actors', 16, 'Number of actor processes to run in parallel.'
)
flags.DEFINE_integer(
    'replay_capacity', 20000, 'Maximum replay size (in number of unrolls stored).'
)   # watch for out of RAM
flags.DEFINE_integer(
    'min_replay_size', 1000, 'Minimum replay size before learning starts (in number of unrolls stored).'
)
flags.DEFINE_bool(
    'clip_grad', True, 'Clip gradients, default on.'
)
flags.DEFINE_float(
    'max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.'
)

flags.DEFINE_float(
    'learning_rate', 0.0001, 'Learning rate for adam.'
)
flags.DEFINE_float(
    'adam_eps', 0.0001, 'Epsilon for adam.'
)
flags.DEFINE_float(
    'discount', 0.997, 'Discount rate.'
)
flags.DEFINE_float(
    'unroll_length', 80, 'Sequence of transitions to unroll before add to replay.'
)
flags.DEFINE_integer(
    'burn_in',
    40,
    'Sequence of transitions used to pass RNN before actual learning.',
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer(
    'batch_size', 32, 'Batch size for learning.'
)

flags.DEFINE_float(
    'priority_exponent', 0.9, 'Priority exponent used in prioritized replay.'
)
flags.DEFINE_float(
    'importance_sampling_exponent', 0.6, 'Importance sampling exponent value.'
)
flags.DEFINE_float(
    'normalize_weights', True, 'Normalize sampling weights in prioritized replay.'
)

flags.DEFINE_float(
    'priority_eta', 0.9, 'Priority eta to mix the max and mean absolute TD errors.'
)
flags.DEFINE_float(
    'rescale_epsilon', 0.001, 'Epsilon used in the invertible value rescaling for n-step targets.'
)
flags.DEFINE_integer(
    'n_step', 5, 'TD n-step bootstrap.'
)

flags.DEFINE_integer(
    'num_iterations', 100, 'Number of iterations to run.'
)
flags.DEFINE_integer(
    'num_train_step', 
    int(5e5), 
    'Number of training steps (environment steps or frames) to run per iteration, per actor.',
)
flags.DEFINE_integer(
    'num_eval_steps', 
    int(2e4), 
    'Number of evaluation steps (environment steps or frames) to run per iteration.',
)
flags.DEFINE_integer(
    'max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.'
)
flags.DEFINE_integer(
    'target_net_update_interval',
    1500,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer(
    'actor_update_interval',
    400,
    'The frequency (measured in actor steps) to update actor local Q network.',
)
flags.DEFINE_float(
    'eval_exploration_epsiolon', 
    0.01,
    'Fixed exploration rate in e-greedy policy for evaluation.',
)
flags.DEFINE_integer(
    'seed',
    1,
    'Runtime seed.',
)
flags.DEFINE_bool(
    'use_tensorboard',
    True,
    'Use Tensorboard to monitor statistics, default on.',
)
flags.DEFINE_bool(
    'actors_on_gpu', 
    True,
    'Run actors on GPU, default on.',
)
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string(
    'tag',
    '',
    'Add tag to Tensorboard log file.',
)
flags.DEFINE_string(
    'results_csv_path',
    './logs/r2d2_atari_results.csv',
    'Path for csv log file.',
)
flags.DEFINE_string(
    'checkpoint_dir',
    './checkpoints',
    'Path for checkpoint directory.',
)

flags.register_validator(
    'environment_frame_stack',
    lambda x: x == 1
)


