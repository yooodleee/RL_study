
"""
Logged Replay Buffer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
from concurrent import futures


from absl import logging
from dopamine.replay_memory import circular_replay_buffer


import gin
import numpy as np
import tensorflow as tf     # compat to version 1.x


gfile = tf.compat.v1.gfile


STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX



class FixedReplayBuffer(object):
    """
    Object composed of a list of OutofGraphReplayBuffers.
    """

    def __init__(
            self,
            data_dir,
            replay_suffix,
            *args,
            replay_file_start_index=0,
            replay_file_end_index=None,
            **kwargs):
        
        """
        Initialize the FixedReplayBuffer class.


        Args:
        -------------
            data_dir: (str)
                log dir from which to load the replay buffer.
            replay_suffix: (int)
                If not None, then only load the replay buffer corresponding
                to the specific suffix in data dir.
            *args:
                Arbatrary extra args.
            replay_file_start_index: (int)
                Starting index of the replay buffer to use.
            replay_file_end_index: (int)
                End index of the replay buffer to use.
            **kwargs:
                Arbtrary keyword args.
        """

        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir

        self._loaded_buffers = False
        self.add_count = np.array(0)

        self._replay_suffix = replay_suffix
        self._replay_indices = self._get_checkpoint_suffixes(
            replay_file_start_index,
            replay_file_end_index
        )

        while not self._loaded_buffers:
            if replay_suffix:
                assert replay_suffix >= 0, \
                'Please pass a non-negative replay suffix'

                self.load_single_buffer(replay_suffix)
            else:
                self._load_replay_buffers(num_buffers=1)
    

    def load_single_buffer(self, suffix):
        """
        Load a single replay buffer.
        """

        replay_buffer = self._loaded_buffers(suffix)
        if replay_buffer is not None:
            
            self._replay_buffers = [replay_buffer]
            self.add_count = replay_buffer.add_count
            self._num_replay_buffers = 1
            self._loaded_buffers = True
    

    def _load_buffer(self, suffix):
        """
        Loads a OutOfGraphReplayBuffer replay buffer.
        """
        try:
            logging.info(
                'Starting to load from ckpt %s',
                suffix,
                self._data_dir
            )
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                *self._args, **self._kwargs
            )
            replay_buffer.load(self._data_dir, suffix)

            replay_capacity = replay_buffer._replay_capacity
            logging.info(
                'Capacity: %d',
                replay_buffer._replay_capacity
            )

            for name, array in replay_buffer._store.items():
                # this frees unused RAM if replay_capacity is smaller than 1M

                replay_buffer._store[name] = array[:replay_capacity + 4].copy()
                logging.info(
                    '%s: %s', name, array.shape
                )
            
            logging.info(
                'Loaded replay buffer ckpt %s from %s',
                suffix, self._data_dir
            )

            return replay_buffer
        

        except tf.errors.NotFoundError:
            return None
    

    def _get_checkpoint_suffixes(
            self,
            replay_file_start_index,
            replay_file_end_index):
        
        """
        Get replay buffer indices to be sampled among all replay buffers.
        """

        ckpts = gfile.ListDirectory(self._data_dir)

        # assumes saved in a format CKPT_NAME.{SUFFIX}.gz
        ckpt_counters = collections.Counter(
            [
                name.split('.')[-2]
                for name in ckpts
                if name.endswith('gz')
            ]
        )

        # should contain the files for add_count, act, obs, reward,
        # terminal and invalid_range
        ckpt_suffixes = [
            int(x) for x in ckpt_counters
            if ckpt_counters[x] in [6, 7]
        ]

        # sort the replay buffer indices. would correspond to list of
        # indices ranging from [0, 1, 2, ...]
        ckpt_suffixes = sorted(ckpt_suffixes)

        if replay_file_end_index is None:
            replay_file_end_index = len(ckpt_suffixes)
        
        replay_indices = ckpt_suffixes[
            replay_file_start_index:replay_file_end_index
        ]

        logging.info(
            'Replay indices: %s', str(replay_indices)
        )

        if len(replay_indices) == 1:
            self._replay_suffix = replay_indices[0]
        
        return replay_indices
    

    def _load_replay_buffers(self, num_buffers):
        """
        Loads multiple checkpoints into a list of replay buffers.
        """

        if not self._loaded_buffers:
            
            ckpt_suffixes = np.random.choice(
                self._replay_indices,
                num_buffers,
                replace=False
            )
            self._replay_buffers = []

            # load the replay buffers in parallel
            with futures.ThreadPoolExecutor(
                max_workers=num_buffers
            ) as thread_pool_executor:
                
                replay_futures = [
                    thread_pool_executor.submit(
                        self._load_buffer,
                        suffix
                    )
                    for suffix in ckpt_suffixes
                ]

            for f in replay_futures:
                replay_buffer = f.result()

                if replay_buffer is not None:
                    self._replay_buffers.append(replay_buffer)
                    self.add_count = max(replay_buffer.add_count, self.add_count)
                
            self._num_replay_buffers = len(self._replay_buffers)
            
            if self._num_replay_buffers:
                self._loaded_buffers = True
    

    def get_transition_elements(self):
        return self._replay_buffers[0].get_transition_elements()
    

    def sample_transition_batch(
            self,
            batch_size=None,
            indices=None):
        
        buffer_index = np.random.randint(
            self._num_replay_buffers
        )

        return self._replay_buffers[buffer_index].sample_transition_batch(
            batch_size=batch_size, indices=indices
        )
    

    def load(self, *args, **kwargs):
        pass


    def reload_buffer(self, num_buffers):

        if not self._replay_suffix:
            self._loaded_buffers = False
            self._load_replay_buffers(num_buffers)
    

    def save(self, *args, **kwargs):
        pass


    def add(self, *args, **kwargs):
        pass



@gin.configurable(
    denylist=[
        'observation_shape',
        'stack_size',
        'update_horizon',
        'gamma',
    ]
)
class WrappedFixedReplayBuffer(
    circular_replay_buffer.WrappedReplayBuffer):

    """
    Wrapper of OutOfGraphReplayBuffer with an in graph sampling mechanism.
    """

    def __init__(
            self,
            data_dir,
            replay_suffix,
            observation_shape,
            stack_size,
            use_staging=True,
            replay_capacity=1000000,
            batch_size=32,
            update_horizon=1,
            gamma=0.99,
            wrapped_memory=None,
            max_sample_attempts=1000,
            extra_storate_types=None,
            observation_dtype=np.uint8,
            action_shape=(),
            action_dtype=np.int32,
            reward_shape=(),
            reward_dtype=np.float32):
        
        """
        Initializes WrappedFixedReplayBuffer.
        """

        memory = FixedReplayBuffer(
            data_dir,
            replay_suffix,
            observation_shape,
            stack_size,
            replay_capacity,
            batch_size,
            update_horizon,
            gamma,
            max_sample_attempts,
            extra_storate_types=extra_storate_types,
            observation_dtype=observation_dtype
        )

        super(WrappedFixedReplayBuffer, self).__init__(
            observation_shape,
            stack_size,
            use_staging=use_staging,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            update_horizon=update_horizon,
            gamma=gamma,
            wrapped_memory=memory,
            max_sample_attempts=max_sample_attempts,
            extra_storate_types=extra_storate_types,
            observation_dtype=observation_dtype,
            action_shape=action_shape,
            action_dtype=action_dtype,
            reward_shape=reward_shape,
            reward_dtype=reward_dtype,
        )