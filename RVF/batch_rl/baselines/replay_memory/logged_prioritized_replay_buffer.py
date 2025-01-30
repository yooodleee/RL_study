# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Logged Prioritized Replay Buffer.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import pickle

# more info: https://github.com/google/dopamine
from dopamine.replay_memory import circular_replay_buffer
from dopamine.replay_memory import prioritized_replay_buffer

import gin
import numpy as np
import tensorflow as tf


SROTE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX


class OutOfGraphLoggedPrioritizedReplayBuffer(
    prioritized_replay_buffer.OutOfGraphPrioritizedReplayBuffer
):
    """
    A logged out-of-graph Replay Buffer for Prioritized Experience Replay.

    """

    def __init__(self, log_dir, *args, **kwargs):
        """
        Initializes OutOfGraphLoggedPrioritizedReplayBuffer.

        """
        super(OutOfGraphLoggedPrioritizedReplayBuffer, self).__init__(
            *args, **kwargs
        )
        self._log_count = 0
        self._log_dir = log_dir
        tf.compat.v1.gfile.MakeDirs(self._log_dir)
    

    def add(
            self,
            observation,
            action,
            reward,
            terminal,
            *args):
        
        super(OutOfGraphLoggedPrioritizedReplayBuffer, self).add(
            observation, action, reward, terminal, *args
        )
        # Log the replay buffer every time the replay buffer is filled to capacity.
        cur_size = self.add_count % self._replay_capacity
        if cur_size == self._replay_capacity - 1:
            self._log_buffer()
            self._log_count += 1
    

    def load(
            self,
            checkpoint_dir,
            suffix):
        
        super(OutOfGraphLoggedPrioritizedReplayBuffer, self).load(
            checkpoint_dir, suffix
        )
        self._log_count = self.add_count // self._replay_capacity
    

    def _load_buffer(self):
        """
        This method will save all the replay buffer's state in a single file.

        """
        checkpoint_elements = self._return_checkpointable_elements()
        for attr in checkpoint_elements:
            filename = self._generate_filename(
                self._log_dir, 
                attr, 
                self._log_count
            )
            with tf.compat.v1.gfile.Open(filename, 'wb') as f:
                with gzip.GzipFile(fileobj=f) as outfile:
                    if attr.startswith(SROTE_FILENAME_PREFIX):
                        array_name = attr[len(SROTE_FILENAME_PREFIX):]
                        np.save(
                            outfile,
                            self._store[array_name],
                            allow_pickle=False,
                        )
                    
                    # Some numpy arrays might not be part of storage
                    elif isinstance(self.__dict__[attr], np.ndarray):
                        np.save(
                            outfile,
                            self.__dict__[attr],
                            allow_pickle=False,
                        )

                    else:
                        pickle.dump(self.__dict__[attr], outfile)
            
            tf.compat.v1.logging.info(
                'Replay buffer logged to ckpt {number} in {dir}'.format(
                    number=self._log_count, dir=self._log_dir
                )
            )
    
    