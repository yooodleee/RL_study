
r"""
The entry point for running experiments for collecting replay datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags

from batch_rl.baselines.agents import dqn_agent
from batch_rl.baselines.agents import quantile_agent
from batch_rl.baselines.agents import random_agent
from batch_rl.baselines.run_experiment import LoggedRunner

from dopamine.discrete_domains import run_experiment
from dopamine.discrete_domains import train as base_train
import tensorflow as tf


flags.DEFINE_string('agent_name', 'dqn', 'Name of the agent.')
FLAGS = flags.FLAGS


def create_agent(
        sess,
        environment,
        replay_log_dir,
        summary_writer=None):
    
    """
    Creates a DQN agent.

    Args:
        sess: A 'tf.Session' object for running associated ops.
        environment: An Atari 2600 environment.
        replay_log_dir: Directory to which log the replay buffers periodically.
        summary_writer: A Tensorflow summary writer to pass to the agent
            for in=agent training statistics in Tensorboard.

    Returns:
        A DQN agent with metrics.
    """
    if FLAGS.agent_name == 'dqn':
        agent = dqn_agent.LoggedDQNAgent
    elif FLAGS.agent_name == 'quantile':
        agent = quantile_agent.LoggedQuantileAgent
    elif FLAGS.agent_name == 'random':
        agent = random_agent.RandomAgent
    else:
        raise ValueError(
            '{} is not a valid agent name'.format(FLAGS.agent_name)
        )
    
    return agent(
        sess,
        num_actions=environment.action_space.n,
        replay_log_dir=replay_log_dir,
        summary_writer=summary_writer,
    )


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.INFO
    )
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    # Create the replay log dir.
    replay_log_dir = os.path.join(FLAGS.base_dir, 'replay_logs')
    tf.compat.v1.logging.info(
        'Saving replay buffer data to {}'.format(replay_log_dir)
    )
    creat_agent_fn = functools.partial(
        create_agent, replay_log_dir=replay_log_dir
    )
    runner = LoggedRunner(FLAGS.base_dir, creat_agent_fn)
    runner.run_experiment()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)