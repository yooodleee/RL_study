
r"""
The entry point for running experiments with fixed replay datasets.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os


from absl import app
from absl import flags

from batch_rl.fixed_replay import run_experiment
from batch_rl.fixed_replay.agents import dqn_agent
from batch_rl.fixed_replay.agents import multi_head_dqn_agent
from batch_rl.fixed_replay.agents import quantile_agent
from batch_rl.fixed_replay.agents import rainbow_agent

from dopamine.discrete_domains import run_experiment as base_run_experiment
from dopamine.discrete_domains import train as base_train

import tensorflow as tf # compat to version 1.x



flags.DEFINE_string(
    'agent_name', 'dqn', 'Name of the agent.'
)
flags.DEFINE_string(
    'replay_dir',
    None,
    'Directory from which to load the replay data'
)
flags.DEFINE_string(
    'init_checkpoint_dir',
    None,
    'Directory from which to load the initial checkpoint before training starts.'
)

FLAGS = flags.FLAGS




def create_agent(
        sess,
        environment,
        replay_data_dir,
        summary_writer=None):
    
    """
    Creates a DQN agent.


    Args
    ---------
        sess: (tf.Session)
            for running associated ops.
        environment:
            An Atari 2600 env.
        replay_data_dir:
            Directory to which log the replay buffers periodically.
        summary_writer:
            A TensorFlow summary writer to pass to the agent for
            in-agent training statistics in Tensorboard.


    Returns
    --------------
        A DQN agent with metrics.
    """

    if FLAGS.agent_name == 'dqn':
        agent = dqn_agent.FixedReplayDQNAgent
    
    elif FLAGS.agent_name == 'c51':
        agent = rainbow_agent.FixedReplayRainbowAgent

    elif FLAGS.agent_name == 'quantile':
        agent = quantile_agent.FixedReplayQuantileAgent
    
    elif FLAGS.agent_name == 'multi_head_dqn':
        agent = multi_head_dqn_agent.FixedReplayMultiHeadDQNAgent
    
    else:
        raise ValueError(
            '{} is not a valid agent name'.format(FLAGS.agent_name)
        )
    
    return agent(
        sess,
        num_actions=environment.action_space.n,
        replay_data_dir=replay_data_dir,
        summary_writer=summary_writer,
        init_checkpoint_dir=FLAGS.init_checkpoint_dir,
    )




def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.INFO
    )
    base_run_experiment.load_gin_configs(
        FLAGS.gin_files,
        FLAGS.gin_bindings,
    )

    replay_data_dir = os.path.join(FLAGS.replay_dir, 'replay_logs')
    
    create_agent_fn = functools.partial(
        create_agent,
        replay_data_dir=replay_data_dir,
    )

    runner = run_experiment.FixedReplayRunner(
        FLAGS.base_dir,
        create_agent_fn,
    )
    runner.run_experiment()


    if __name__ == "__main__":
        flags.mark_flag_as_required('replay_dir')
        flags.mark_flags_as_required('base_dir')

        app.run(main)