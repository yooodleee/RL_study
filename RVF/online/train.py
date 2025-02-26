
r"""The entry point for running experiments."""


from absl import app
from absl import flags
from batch_rl.multi_head import multi_network_dqn_agent
from batch_rl.multi_head import quantile_agent
from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import run_experiment
import tensorflow as tf     # compat to version 1.x


flags.DEFINE_string('agent_name', 'dqn', 'Name of the agent.')
flags.DEFINE_string('base_dir', None, 'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").'
)
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1)",'
    '       "create_environment.game_name="Pong"").'
)


FLAGS = flags.FLAGS



def create_agent(sess, environment, summary_writer=None):
    """Creates an online agent.
    
    
    Args
    ---------------
        sess: A `tf.Session` obj for running associated ops.
        environment: An Atari 2600 env.
        summary_writer: A TensorFlow summary writer to pass to the agent for in-agent
            training statistics in Tensorboard.
            
    
    Returns
    ----------------
        A DQN agent with metrics.
    """
    if FLAGS.agent_name == 'dqn':
        agent = dqn_agent.DQNAgent
    
    elif FLAGS.agent_name == 'c51':
        # Gin config ensures that only run C51 component of Rainbow
        agent = rainbow_agent.RainbowAgent
    
    elif FLAGS.agent_name == 'quantile':
        agent = quantile_agent.QuantileAgent
    
    elif FLAGS.agent_name == 'rem':
        agent = multi_network_dqn_agent.MultiNetworkDQNAgent
    
    else:
        raise ValueError('{} is not a valid agent name'.format(FLAGS.agent_name))
    
    return agent(
        sess, num_actions=environment.action_space.n, summary_writer=summary_writer
    )



def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_experiment.Runner(FLAGS.base_dir, create_agent)
    runner.run_experiment()



if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)