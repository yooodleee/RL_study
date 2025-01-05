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
from torch.utils.tensorboard import SummaryWriter

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
    
    def reset(self)-> None:
        """
        Resets all gathered statistics, not to be called between episodes.
        """
        self._num_steps_since_reset = 0
        self._episode_returns = []
        self._episode_steps = []
        self._episode_visited_rooms = []
        self._current_episode_step = 0
        self._current_episode_rewards = []
    
    def get(self)-> Mapping[str, Union[int, float, None]]:
        """
        Aggregates statistics and returns as a dictionary.

        Here the convention is `episode_return` is set to `current_episode_return`
            if a full episode has not been encountered. Otherwise it is set to
            `mean_episode_return` which is the mean return of complete episodes only.
            If no steps have been taken at all, `episode_return` is set to `NaN`.

        Returns:
            A dictionary of aggregated statistics.
        """

        # Note most games don't have visited romms info
        mean_episode_visited_rooms = 0

        if len(self._episode_returns) > 0:
            mean_episode_return = np.array(
                self._episode_returns
            ).mean()

            if len(self._episode_visited_rooms) > 0:
                mean_episode_visited_rooms = np.array(
                    self._episode_visited_rooms
                ).mean()
        else:
            mean_episode_return = sum(self._current_episode_rewards)

        return {
            'mean_episode_return': mean_episode_return,
            'mean_episode_visited_rooms': mean_episode_visited_rooms,
            'num_episodes': len(self._episode_returns),
            'current_episode_step': self._current_episode_step,
            'num_steps_since_reset': self._num_steps_since_reset,
        }


class StepRateTracker:
    """
    Tracks step rate, number of steps taken and duration since last reset.
    """

    def __init__(self):
        self._num_steps_since_reset = None
        self._start = None

    def step(
        self, env, timestep_t, agent, a_t
    )-> None:
        """
        Accumulates statistics from timestep.
        """
        del (env, timestep_t, agent, a_t)

        self._num_steps_since_reset += 1
    
    def reset(self)-> None:
        """
        Reset statistics.
        """
        self._num_steps_since_reset = 0
        self._start = timeit.default_timer()
    
    def get(self)-> Mapping[Text, float]:
        """
        Returns statistics as a dictionary.
        """
        duration = timeit.default_timer() - self._start
        if self._num_steps_since_reset > 0:
            step_rate = self._num_steps_since_reset / duration
        else:
            step_rate = np.nan
        return {
            'step_rate': step_rate,
            'num_steps_since_reset': self._num_steps_since_reset,
            'duration': duration,
        }


class TensorboardEpisodeTracker(EpisodeTracker):
    """
    Extend EpisodeTracker to write to tensorboard.
    """

    def __init__(self, writer: SummaryWriter):
        super().__init__()
        self._total_steps = 0   # Keep track total number of steps, does not reset
        self._total_episodes = 0    # Keep track total number of episodes, does not reset
        self._writer = writer
    
    def step(
        self, env, timestep_t, agent, a_t
    )-> None:
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            self._total_episodes += 1
            tb_steps = self._total_steps

            # tracker per episode
            episode_return = self._episode_returns[-1]
            episode_step = self._episode_steps[-1]

            # tracker per step
            self._writer.add_scalar(
                'performance(env_steps)/num_episodes',
                self._total_episodes,
                tb_steps,
            )
            self._writer.add_scalar(
                'performance(env_steps)/episode_return',
                episode_return,
                tb_steps,
            )
            self._writer.add_scalar(
                'performance(env_steps)/episode_steps',
                episode_step,
                tb_steps,
            )

            # For Atari games like MontezumaRevenge and Pitfall
            if isinstance(
                timestep_t.info, dict
            ) and 'episode_visited_rooms' in timestep_t.info:
                episode_visited_rooms = self._episode_visited_rooms[-1]
                self._writer.add_scalar(
                    'performance(env_steps)/episode_visited_rooms',
                    episode_visited_rooms,
                    tb_steps,
                )


class TensorboardStepRateTracker(StepRateTracker):
    """
    Extend stepRateTracker to write to tensorboard, for single thread training
        agent only.
    """

    def __init__(self, writer: SummaryWriter):
        super().__init__()

        self._total_steps = 0   # Keep track total number of steps, does not reset
        self._writer = writer

    def step(
        self, env, timestep_t, agent, a_t
    )-> None:
        """
        Accumulates statistics from timestep.
        """
        super().step(env, timestep_t, agent, a_t)

        self._total_steps += 1

        # To improve performance, only logging at end of an episode.
        if timestep_t.done:
            time_stats = self.get()
            self._writer.add_scalar(
                'performance(env_steps)/step_rate',
                time_stats['step_rate'],
                self._total_steps,
            )


