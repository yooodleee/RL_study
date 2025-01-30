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
Logged Replay Buffer.
"""

from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function

import gzip
import pickle

from dopamine.replay_memory import circular_replay_buffer

import gin
import numpy as np
import tensorflow as tf


STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX


class OutOfGraphLoggedReplayBuffer(
    circular_replay_buffer.OutOfGraphReplayBuffer
):
    """
    Logs the replay buffer to disk everytime it's full.

    """

    def __init__(self, log_dir, *args, **kwargs):
        super(OutOfGraphLoggedReplayBuffer, self).__init__(*args, **kwargs)

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
        
        super(OutOfGraphLoggedReplayBuffer, self).add(
            observation, action, reward, terminal, *args
        )
        
        # Log the replay buffer to a file in self._log_dir if the replay buffer
        # is full.
        cur_size = self.add_count % self._replay_capacity
        if cur_size == self._replay_capacity - 1:
            self._log_buffer()
            self._log_count += 1
    
    