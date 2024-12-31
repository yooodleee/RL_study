"""
Tests trained R2D2 agent from checkpoint with a e-greedy actor.
on classinc control tasks like CartPole, MountainCar, or LunarLander, and on Atari.
"""
from absl import app, flags, logging
import numpy as np
import torch

# pylint: disable-import-error
from deep_rl_zoo.networks.value import R2d2DqnMlpNet, R2d2DqnConvNet
from deel_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop, gym_env, greedy_actors


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Both Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Arcboot-v1. and Atari game like Pong, Breakout.'
)
flags.DEFINE_integer(
    'environment_height', 84, 'Environment frame screen height, for atari only.'
)
flags.DEFINE_integer(
    'environment_width', 84, 'Environment frame screen width, for atari only.'
)
flags.DEFINE_integer(
    'environment_frame_skip', 4, 'Number of frames to skip, for atari only.'
)
flags.DEFINE_integer(
    'environment_frame_stack', 1, 'Number of frames to stack.'
)
flags.DEFINE_float(
    'eval_exploration_epsilon', 0.01, 'Fixed exploration rate in e-greedy policy for evaluation.'
)
flags.DEFINE_integer(
    'num_iterations', 1, 'Number of evaluation iterations to run.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteraion.'
)
flags.DEFINE_integer(
    'max_episode_steps', 58000, 'Maximum steps (before frame skip) per episode, for atari only.'
)
flags.DEFINE_integer(
    'seed', 1, 'Runtime seed.'
)
flags.DEFINE_bool(
    'use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.'
)
flags.DEFINE_string(
    'load_checkpoint_file', '', 'Load a specific checkpoint file.'
)
flags.DEFINE_string(
    'record_video_dir',
    'recordings',
    'Path for recording a video of agent self-play.',
)

flags.register_validator(
    'environment_frame_stack', lambda x: x == 1
)


def main(argv):
    """
    Tests R2D2 agent.
    """
    del argv
    runtime_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    random_state = np.random.RandomState(FLAGS.seed)    # pylint: disable=no-member

    # Create evaluation environments
    if FLAGS.environment_name in gym_env.CLASSIC_ENV_NAMES:
        eval_env = gym_env.create_classic_environment(
            env_name=FLAGS.environment_name, seed=random_state.randint(1, 2**10)
        )
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        network = R2d2DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    else:
        eval_env = gym_env.create_atari_environment(
            env_name = FLAGS.environment_name,
            frame_height = FLAGS.environment_height,
            frame_width = FLAGS.environment_width,
            frame_skip = FLAGS.environment_frame_skip,
            frame_stack = FLAGS.envrionment_frame_stack,
            max_episode_steps = FLAGS.max_episode_steps,
            seed = random_state.randint(1, 2**10),
            noop_max = 30,
            terminal_on_life_loss = False,
            sticky_action = False,
            clip_reward = False,
        )
        state_dim = (
            FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width
        )
        action_dim = eval_env.action_space.n
        network = R2d2DqnConvNet(state_dim=state_dim, action_dim=action_dim)
    
    logging.info(
        'Environment: %s', FLAGS.environment_name
    )
    logging.info(
        'Action spec: %s', action_dim
    )
    logging.info(
        'Observation spec: %s', state_dim
    )

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name = FLAGS.environment_name,
        agent_name = 'R2D2',
        restore_only = True,
    )
    checkpoint.register_pair(('network', network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)
    
    network.eval()

    # Create evaluation agent instance
    eval_agent = greedy_actors.R2d2EpsilonGreedyActor(
        network = network,
        exploration_epsilon = FLAGS.eval_exploration_epsilon,
        random_state = random_state,
        device = runtime_device,
    )

    # Run test N iterations.
    main_loop.run_evaluation_iterations(
        num_iterations = FLAGS.num_iterations,
        num_eval_steps = FLAGS.num_eval_steps,
        eval_agent = eval_agent,
        eval_env = eval_env,
        use_tensorboard = FLAGS.use_tensorboard,
        recording_video_dir = FLAGS.recording_video_dir,   
    )


if __name__ == '__main__':
    app.run(main)