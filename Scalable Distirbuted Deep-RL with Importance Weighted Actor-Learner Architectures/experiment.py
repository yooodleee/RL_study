"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow_addons import rnn

import collections
import contextlib
import functools
import os
import sys

import dmlab30
import environments
import numpy as np
import py_process
import sonnet as snt
import tensorflow as tf
import vtrace

try:
    import dynamic_batching
except tf.errors.NotFoundError:
    tf.compat.v1.logging.warning('Running without dynamic batching.')

from six.moves import range


# nest = tf.contrib.framework.nest
nest = tf.nest

# flags = tf.app.flags
flags = tf.compat.v1.flags
# FLAGS = tf.app.flags.FLAGS
FLAGS = tf.compat.v1.flags.FLAGS

flags.DEFINE_string('logidr', '/tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

# Flags used for testing.
flags.DEFINE_integer('test_num_episodes', 10, 'Number of episodes per level.')

# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum(
    'job_name', 'learner', ['learner', 'actor'],
    'Job name. Ignored when task is set to -1.')

# Training.
flags.DEFINE_integer(
    'total_environment_frames', int(1e9),
    'Total environment frames to train for.')
flags.DEFINE_integer('num_actors', 4, 'Number of actors.')
flags.DEFINE_integer('batch_size', 2, 'Batch size for training.')
flags.DEFINE_integer('unroll_length', 100, 'Unroll length in agent steps.')
flags.DEFINE_integer('num_action_repeats', 4, 'Number of action repeats.')
flags.DEFINE_integer('seed', 1, 'Random seed.')

# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float(
    'reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'], 'Reward clipping.')

# Environment settings.
flags.DEFINE_string(
    'dataset_path', '',
    'Path to dataset needed for psychlab_*, see '
    'https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008')
flags.DEFINE_string(
    'level_name', 'explore_goal_locations_small',
    '''Level name or \'dmlab30\' for the full DmLab-30 suite '''
    '''with levels assigned round robin to the actors.''')
flags.DEFINE_integer('width', 96, 'Width of observation.')
flags.DEFINE_integer('height', 72, 'Height of observation.')

# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.00048, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name_agent_state_env_outputs agent_outputs')
AgentOutput = collections.namedtuple(
    'AgentOutput', 'action policy_logits baseline.')


def is_single_machine():
    return FLAGS.task == -1


class Agent(snt.RNNCore):
    """Agent with ResNet."""

    def __init__(self, num_actions):
        super(Agent, self).__init__(name='agent')

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.keras.layers.LSTMCell(256)
    
    def initial_state(self, batch_size):
        return self._core.zero_state(batch_size, tf.float32)
    
    def _instruction(self, instruction):
        # Split string.
        splitted = tf.compat.v1.string_split(instruction)
        dense = tf.compat.v1.sparse_tensor_to_dense(splitted, default_value='')
        length = tf.reduce_sum(
            tf.compat.v1.to_int32(tf.not_equal(dense, '')), axis=1)