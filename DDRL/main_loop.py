"""
Training loops.
"""

from typing import (
    Iterable, List, Tuple, Text, Mapping, Any
)
import itertools
import collections
import sys
import time
import signal
import queue
import math
import multiprocessing
import threading
from absl import logging
import gym

# pylint: disable=import-error
import trackers as trackers_lib
from . import types as types_lib
from log import CsvWriter
from checkpoint import PyTorchCheckpoint
import gym_env


def run_env_loop(
    agent: types_lib.Agent, env: gym.Env
)-> Iterable[Tuple[
    gym.Env,
    types_lib.TimeStep,
    types_lib.Agent,
    types_lib.Action,
]]:
    """
    Repeatedly alternates step calls on environemtn and agent.

    At time `t`, `t+1` environemtn timesteps and `t+1` agent steps have been
        seen in the current episode. `t` resets to `0` for the next episode.

    Args:
        agent: Agent to be run, has methods `step(timestep)` and `reset()`.
        env: Environment to run, has methods `step(action)` and `reset()`.

    Yields:
        Tuple `(env, timestep_t, agent, a_t)` where
            `a_t = agent.step(timestep_t)`.

    Raises:
        RuntimeError if the `agent` is not an instance of types_lib.Agent.
    """

    if not isinstance(agent, types_lib.Agent):
        raise RuntimeError(
            'Expect agent to be an instance of types_lib.Agent.'
        )
    
    while True:
        # For each episode.
        agent.reset()
        # Think of reset as a special 'action' the agent takes, thus given us a reward 
        # 'zero', and a new state 's_t'.
        observation = env.reset()
        reward = 0.0
        done = loss_life = False
        first_step = True
        info = {}

        while True:
            # For each step in the current episode.
            timestep_t = types_lib.TimeStep(
                observation=observation,
                reward=reward,
                done=done or loss_life,
                first=first_step,
                info=info,
            )
            a_t = agent.step(timestep_t)
            yield env, timestep_t, agent, a_t

            a_tm1 = a_t
            observation, reward, done, info = env.step(a_tm1)

            # For Atari games, check if should treat loss a life as a 
            # short-terminal state
            loss_life = False
            if 'loss_life' in info and info['loss_life']:
                loss_life = info['loss_life']
            
            if done:
                # Actual end of an episode
                # This final agent.step() will ensure the done state and final reward
                # will be seen by the agent and the trackers
                timestep_t = types_lib.TimeStep(
                    observation=observation,
                    reward=reward,
                    done=True,
                    first=False,
                    info=info,
                )
                unused_a = agent.step(timestep_t)   # noqa: F841
                yield env, timestep_t, agent, None
                break


def run_env_steps(
    num_steps: int,
    agent: types_lib.Agent,
    env: gym.Env,
    trackers: Iterable[Any],
)-> Mapping[Text, float]:
    """
    Run some stps and return the statistics, this could be either training,
        evaluation, or testing steps.

    Args:
        max_episode_steps: maximum steps per episode.
        agent: agent to run, expect the agent to have step(), reset(),
            and a agent_name property.
        train_env: training environment.
        trackers: statistics trackers.

    Returns:
        A Dict contains statistics about the result.
    """
    seq = run_env_loop(agent, env)
    seq_truncated = itertools.islice(seq, num_steps)
    stats = trackers_lib.generate_statistics(trackers, seq_truncated)

    return stats


