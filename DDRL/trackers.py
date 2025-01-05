"""
Components for statistics and Tensorboard monitoring.
"""

import timeit
from pathlib import Path
import shutil
import collections
from typing import (
    Any,
    Text,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)
import numpy as np
from torch.utils.tensorboard import summary

# pylint: disable=model-error
import replay as replay_lib


class EpisodeTracker:
    """
    Tracks episode return and other statistics.
    """

    def __init__(self):
        self._num_steps_since_reset = None
        self._episode_returns = None
        self._episode_steps = None
        self._episode_visited_rooms = None
        self._current_episode_rewards = None
        self._current_episode_step = None
    
    def step(
        self,
        env,
        timestep_t,
        agent,
        a_t,
    )-> None:
        """
        Accumulates statistics from timestep.
        """
        del(env, agent, a_t)

        # First reward is invalid, all other rewards are appended.
        if timestep_t.first:
            if self._current_episode_rewards:
                raise ValueError(
                    'Current episode reward list should be empty.'
                )
            if self._current_episode_step != 0:
                raise ValueError(
                    'Current episode step should be zero.'
                )
        else:
            reward = timestep_t.reward
        
            # Try to use the non-clipped/unscaled raw reward when collecting statistics
            if isinstance(
                timestep_t.info, dict
            ) and 'raw_reward' in timestep_t.info:
                reward = timestep_t.info['raw_reward']
            
            self._current_episode_rewards.append(reward)

        self._num_steps_since_reset += 1
        self._current_episode_step += 1

        if timestep_t.done:
            self._episode_returns.append(
                sum(self._current_episode_rewards)
            )
            self._episode_steps.append(self._current_episode_step)
            self._current_episode_rewards = []
            self._current_episode_step = 0

            # For Atari games like MontezumaRevenge and Pitfall
            if isinstance(
                timestep_t.info, dict
            ) and 'episode_visited_rooms' in timestep_t.info:
                self._episode_visited_rooms.append(
                    timestep_t.info['episode_visited_rooms']
                )
    
    