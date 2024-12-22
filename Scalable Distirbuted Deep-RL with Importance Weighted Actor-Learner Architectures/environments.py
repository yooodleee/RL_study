"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path

import numpy as np
import tensorflow as tf

import deepmind_lab


# nest = tf.contrib.framework.nest
nest = tf.nest


class LocalLevelCache(object):
    """Local level cache."""

    def __init__(self, cache_dir='/tmp/level_cache'):
        self._cache_dir = cache_dir
        self.gfile.MakeDirs(cache_dir)
    
    def fetch(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if tf.compat.v1.gfile.Exists(path):
            tf.compat.v1.gfile.Copy(path, pk3_path, overwrite=True)
            return True
        return False
    
    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if not tf.compat.v1.gfile.Exists(path):
            tf.compat.v1.gfile.Copy(pk3_path, path)


DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),      # Forward
    (0, 0, 0, -1, 0, 0, 0),     # Backward
    (0, 0, -1, 0, 0, 0, 0),     # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),      # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),    # Look Left
    (20, 0, 0, 0, 0, 0, 0),     # Look Right
    (-20, 0, 0, 1, 0, 0, 0),    # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),     # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),      # Fire.
)


class PyProcessDmLab(object):
    """DeepMind Lab wrapper for PyProcess."""

    def __init__(
        self,
        level,
        config,
        num_action_repeats,
        seed,
        runfiles_path=None,
        level_cache=None
    ):
        self._num_action_repeats = num_action_repeats
        self._random_state = np.random.RandomState(seed=seed)
        if runfiles_path:
            deepmind_lab.set_runfiles_path(runfiles_path)
        config = {k: str(v) for k, v in config.iteritems()}
        self._observation_spec = ['RGB_INTERLEAVED', 'INSTR']
        self._env = deepmind_lab.Lab(
            level=level,
            observations=self._observation_spec,
            config=config,
            level_cache=level_cache,
        )
    
    def _reset(self):
        self._env.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
    
    def _observation(self):
        d = self._env.observations()
        return [d[k] for k in self._observation_spec]
    
    def initial(self):
        self._reset()
        return self._observation()
    
    def step(self, action):
        reward = self._env.step(action, num_steps=self._num_action_repeats)
        done = np.array(not self._env.is_running())
        if done:
            self._reset()
        observation = self._observation()
        reward = np.array(reward, dtype=np.float32)
        return reward, done, observation
    
    def close(self):
        self._env.close()
    
    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of 'TensorSpec' with the method's output specification."""
        width = constructor_kwargs['config'].get('width', 320)
        height = constructor_kwargs['config'].get('height', 240)

        observation_spec = [
            tf.TensorSpec([height, width, 3], tf.unit8),
            tf.TensorSpec([], tf.string),
        ]

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            return (
                tf.TensorSpec([], tf.float32),
                tf.TensorSpec([], tf.bool),
                observation_spec,
            )


StepOutputInfo = collections.namedtuple(
    'StepOutputInfo', 'episode_return_episode_step')

StepOutput = collections.namedtuple(
    'StepOutput', 'reward info done observation')


class FlowEnvironment(object):
    """
    An environment that returns a new state for every modifying method.

    The environment returns a new environment state for every modifying action and
    forces previous actions to be completed first. Similar to 'flow' for
    'TensorArray'.
    """

    def __init__(self, env):
        """
        Initializes the environment.

        Args:
            env: An environment with 'initial()' and 'step(action)' methods where
                'initial' returns the initial observations and 'step' takes an action
                and returns a tuple of (reward, done, observation). 'observation'
                should be the observation after the step is taken. If 'done' is
                True, the observation should be the first observation in the next episode.
        """
        self._env = env
    
    def initial(self):
        """
        Returns the initial output and initial state.

        Returns:
            A tuple of ('StepOutput', environment state). The environment state should
            be passed in to the next invocation of 'step' and should not be used in 
            any other way. The reward and transition type in the 'StepOutput' is the
            reward/transition type that lead to the observation in 'StepOutput'.
        """
        with tf.name_scope('flow_environment_initial'):
            ini