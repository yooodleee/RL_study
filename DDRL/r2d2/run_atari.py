"""
From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning
https://openreview.net/pdf?id=r1lyTjAqYX.
"""

from absl import app, flags, logging
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import multiprocessing
import numpy as np
import torch
import copy

# pylint: disable=import-error
from networks.value import R2d2DqnConvNet, RnnDqnNetworkInputs
from r2d2 import agent
from checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop, gym_env, greedy_actors
from deel_rl_zoo import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name', 
    'Pong', 
    'Atari name without NoFrameskip and version, like Breakout, Pong, Sequest.'
)
flags.DEFINE_integer(
    'environment_height', 
    84, 
    'Environment frame screen height.'
)
flags.DEFINE_integer(
    'environment_width', 
    84, 
    'Environment frame screen width.'
)
flags.DEFINE_integer(
    'environment_frame_skip', 
    4, 
    'Number of frames to skip.'
)
flags.DEFINE_integer(
    'environment_frame_stack', 
    1, 
    'Number of frames to stack.'
)
flags.DEFINE_bool(
    'compress_state', 
    True, 
    'Compress state images when store in experience replay.'
)
flags.DEFINE_integer(
    'num_actors', 
    16, 
    'Number of actor processes to run in parallel.'
)
flags.DEFINE_integer(
    'replay_capacity', 
    20000, 
    'Maximum replay size (in number of unrolls stored).'
)   # watch for out of RAM
flags.DEFINE_integer(
    'min_replay_size', 
    1000, 
    'Minimum replay size before learning starts (in number of unrolls stored).'
)
flags.DEFINE_bool(
    'clip_grad', 
    True, 
    'Clip gradients, default on.'
)
flags.DEFINE_float(
    'max_grad_norm', 
    40.0, 
    'Max gradients norm when do gradients clip.'
)

flags.DEFINE_float(
    'learning_rate', 
    0.0001, 
    'Learning rate for adam.'
)
flags.DEFINE_float(
    'adam_eps', 
    0.0001, 
    'Epsilon for adam.'
)
flags.DEFINE_float(
    'discount', 
    0.997, 
    'Discount rate.'
)
flags.DEFINE_float(
    'unroll_length', 
    80, 
    'Sequence of transitions to unroll before add to replay.'
)
flags.DEFINE_integer(
    'burn_in',
    40,
    'Sequence of transitions used to pass RNN before actual learning.',
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two consecutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer(
    'batch_size', 
    32, 
    'Batch size for learning.'
)

flags.DEFINE_float(
    'priority_exponent', 
    0.9, 
    'Priority exponent used in prioritized replay.'
)
flags.DEFINE_float(
    'importance_sampling_exponent', 
    0.6, 
    'Importance sampling exponent value.'
)
flags.DEFINE_float(
    'normalize_weights', 
    True, 
    'Normalize sampling weights in prioritized replay.'
)

flags.DEFINE_float(
    'priority_eta', 
    0.9, 
    'Priority eta to mix the max and mean absolute TD errors.'
)
flags.DEFINE_float(
    'rescale_epsilon', 
    0.001, 
    'Epsilon used in the invertible value rescaling for n-step targets.'
)
flags.DEFINE_integer(
    'n_step', 
    5, 
    'TD n-step bootstrap.'
)

flags.DEFINE_integer(
    'num_iterations', 
    100, 
    'Number of iterations to run.'
)
flags.DEFINE_integer(
    'num_train_step', 
    int(5e5), 
    'Number of training steps (environment steps or frames) to run per iteration, per actor.',
)
flags.DEFINE_integer(
    'num_eval_steps', 
    int(2e4), 
    'Number of evaluation steps (environment steps or frames) to run per iteration.',
)
flags.DEFINE_integer(
    'max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.'
)
flags.DEFINE_integer(
    'target_net_update_interval',
    1500,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer(
    'actor_update_interval',
    400,
    'The frequency (measured in actor steps) to update actor local Q network.',
)
flags.DEFINE_float(
    'eval_exploration_epsiolon', 
    0.01,
    'Fixed exploration rate in e-greedy policy for evaluation.',
)
flags.DEFINE_integer(
    'seed',
    1,
    'Runtime seed.',
)
flags.DEFINE_bool(
    'use_tensorboard',
    True,
    'Use Tensorboard to monitor statistics, default on.',
)
flags.DEFINE_bool(
    'actors_on_gpu', 
    True,
    'Run actors on GPU, default on.',
)
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string(
    'tag',
    '',
    'Add tag to Tensorboard log file.',
)
flags.DEFINE_string(
    'results_csv_path',
    './logs/r2d2_atari_results.csv',
    'Path for csv log file.',
)
flags.DEFINE_string(
    'checkpoint_dir',
    './checkpoints',
    'Path for checkpoint directory.',
)

flags.register_validator(
    'environment_frame_stack',
    lambda x: x == 1
)


def main(argv):
    """
    Trains R2D2 agent on Atari.
    """
    del argv

    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs R2D2 agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    random_state = np.random.RandomState(FLAGS.seed)    # pylint: disable=no-member

    # Create evaluation envrionment, like R2D2, we disable terminate-on-life-loss and clip reward.
    def environment_builder():
        return gym_env.create_atari_environment(
            env_name = FLAGS.environment_name,
            frame_height = FLAGS.environment_height,
            frame_width = FLAGS.environment_width,
            frame_skip = FLAGS.environment_frame_skip,
            frame_stack = FLAGS.environment_frame_stack,
            max_episode_steps = FLAGS.max_episode_steps,
            seed = random_state.randint(1, 2**10),
            noop_max = 30,
            terminal_on_life_loss = True,
            sticky_action = False,
            clip_reward = False,
        )
    
    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape
    action_dim = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Test environment and state shape.
    obs = eval_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (
        FLAGS.environment_frame_stack,
        FLAGS.environment_height,
        FLAGS.environment_width,
    )

    # Create network for learner to optimize, actor will use the same network with share memory.
    network = R2d2DqnConvNet(
        state_dim=state_dim, action_dim=action_dim
    )
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr = FLAGS.learning_rate,
        eps = FLAGS.adam_eps,
    )

    # Test network output.
    x = RnnDqnNetworkInputs(
        s_t = torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1 = torch.zeros(1, 1).long(),
        r_t = torch.zeros(1, 1).float(),
        hidden_s = network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    assert network_output.q_values.shape == (1, 1, action_dim)
    assert len(network_output.hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent
    
    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_t = replay_lib.compress_array(transition.s_t),
            )
        
        def decoder(transition):
            return transition._replace(
                s_t = replay_lib.uncompress_array(transition.s_t),
            )
    
    else:
        encoder = None
        decoder = None
    
    replay = replay_lib.Prioritizedreplay(
        capacity = FLAGS.replay_capacity,
        structure = agent.TransitionsStructure,
        priority_exponent = FLAGS.priority_exponent,
        importance_sampling_exponent = importance_sampling_exponent_schedule(),
        normalize_weights = FLAGS.normalize_weights,
        random_state = random_state,
        time_major = True,
        encode = encoder(),
        decoder = decoder(),
    )

    # Create queue to shared transitions between actors and learner
    data_queue = multiprocessing.Queue(
        maxsize=FLAGS.num_actors * 2
    )

    # Create shared objects so all actors processes can access them
    manager = multiprocessing.Manager()

    # Store copy of latest parameters of the nn in a shared directory,
    # so actors can later access it
    shared_params = manager.dict({'network': None})

    # Create R2D2 learner instance
    laerner_agent = agent.Learner(
        network = network,
        optimizer = optimizer,
        replay = replay,
        min_replay_size = FLAGS.min_replay_size,
        target_net_update_interval = FLAGS.target_net_update_interval,
        discount = FLAGS.discount,
        burn_in = FLAGS.burn_in,
        priority_eta = FLAGS.priority_eta,
        rescale_epsilon = FLAGS.rescale_epsilon,
        batch_size = FLAGS.batch_size,
        n_step = FLAGS.n_step,
        clip_grad = FLAGS.clip_grad,
        max_grad_norm = FLAGS.max_grad_norm,
        device = runtime_device,
        shared_params = shared_params,
    )

    # Create actor environment, actor instances.
    actor_envs = [
        environment_builder() for _ in range(FLAGS.num_actors)
    ]

    actor_diveces = ['cpu'] * FLAGS.num_actors
    # Evenly distribute the actors to all available GPUs
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_diveces = [
            torch.device(f'cuda:{i % num_gpus}')
            for i in random_state(FLAGS.num_actors)
        ]
    
    # Rank 0 is the most explorative actor, while rank N-1 is the most exploitative actor.
    # Each actor has it's own network with different weights.
    actors = [
        agent.Actor(
            rank = 1,
            data_queue = data_queue,
            network = copy.deepcopy(network),
            random_state = np.random.RandomState(FLAGS.seed + int(i)),  # pylint: disable=no-member
            num_actors = FLAGS.num_actors,
            action_dim = action_dim,
            unroll_length = FLAGS.unroll_length,
            burn_in = FLAGS.burn_in,
            actor_update_interval = FLAGS.actor_update_interval,
            device = actor_diveces[i],
            shared_params = shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.R2d2EpsilonGreedActor(
        network = network,
        exploration_epsilon = FLAGS.eval_exploration_epsilon,
        random_state = random_state,
        device = runtime_device,
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name = FLAGS.environment_name,
        agent_name = 'R2D2',
        save_dir = FLAGS.checkpoint_dir,
    )
    checkpoint.register_pair(('network', network))

    # Run parallel training iterations.
    main_loop.run_parallel_training_iterations(
        num_iterations = FLAGS.num_iterations,
        num_train_steps = FLAGS.num_train_steps,
        num_eval_steps = FLAGS.num_eval_steps,
        laerner_agent = laerner_agent,
        eval_agent = eval_agent,
        eval_env = eval_env,
        actors = actors,
        actor_envs = actor_envs,
        data_queue = data_queue,
        checkpoint = checkpoint,
        csv_file = FLAGS.results_csv_path,
        use_tensorboard = FLAGS.use_tensorboard,
        tag = FLAGS.tag,
        debug_screenshots_interval = FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)