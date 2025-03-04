"""
From the paper "Recurrent Experience Replay in Distributed Reinforcement Learning"
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


from networks.value import R2d2DqnMlpNet, RnnDqnNetworkInputs
from r2d2 import agent
from checkpoint import PyTorchCheckpoint
from .. import (
    main_loop,
    gym_env,
    greedy_actors,
)
from .. import replay as replay_lib


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrbot-v1.',
)
flags.DEFINE_string(
    'num_actors',
    16,
    'Number of actor proceses to use.',
)
flags.DEFINE_integer(
    'replay_capacity',
    10000,
    'Maximum_replay_size (in number of unrolls stored).',
)
flags.DEFINE_integer(
    'min_replay_size',
    1000,
    'Maximum replay size before learning starts (in number of unrolls stored).',
)
flags.DEFINE_bool(
    'clip_grad',
    True,
    'Clip gradients, default on.',
)
flags.DEFINE_float(
    'max_grad_norm',
    0.5,
    'Max gradients norm when do gradients clip.',
)

flags.DEFINE_float(
    'learning_rate',
    0.0005,
    'Learning rate for adam.',
)
flags.DEFINE_float(
    'adam_eps',
    0.001,
    'Epsilon for adam.',
)
flags.DEFINE_float(
    'discount',
    0.997,
    'Discount rate.',
)
flags.DEFINE_integer(
    'unroll_length',
    15,
    'Sequence of transitions to unroll before add to replay.',
)
flags.DEFINE_integer(
    'burn_in',
    0,
    'Sequence of transitions used to pass RNN before actual learning.',
    'The effective length of unrolls will be burn_in + unroll_length, '
    'two conseucutive unrolls will overlap on burn_in steps.',
)
flags.DEFINE_integer(
    'batch_size',
    32,
    'Batch size for laerning.',
)

flags.DEFINE_float(
    'priority_exponent',
    0.9,
    'Priority exponent used in prioritized replay.',
)
flags.DEFINE_float(
    'importance_sampling_exponent',
    0.6,
    'Importance sampling exponent value.',
)
flags.DEFINE_bool(
    'normalize_weights',
    True,
    'Normalize sampling weights in prioritized replay.',
)
flags.DEFINE_float(
    'priority_eta',
    0.9,
    'Priority eta to mix the max and mea absolute TD errors.',
)
flags.DEFINE_float(
    'rescale_epsilon',
    0.001,
    'Epsilon used in the invertible value rescaling for n-step targets.',
)
flags.DEFINE_integer(
    'n_step',
    5,
    'TD n-step bootstrap.',
)

flags.DEFINE_integer(
    'num_iterations',
    2,
    'Number of iterations to run.',
)
flags.DEFINE_integer(
    'num_train_steps',
    int(5e5),
    'Number of training env steps to run per iteration, per actor.',
)
flags.DEFINE_integer(
    'num_eval_steps',
    int(2e4),
    'Number of evaluation env steps to run per iteration.',
)
flags.DEFINE_integer(
    'target_net_update_interval',
    100,
    'The interval (meassured in Q network updates) to update target Q networks.',
)
flags.DEFINE_integer(
    'actor_update_interval',
    100,
    'The frequency (measured in actor steps) to update actor local Q network.',
)
flags.DEFINE_float(
    'eval_exploration_epsilon',
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
    './logs/r2d2_classic_results.csv',
    'Path for CSV log file.',
)
flags.DEFINE_string(
    'checkpoint_dir',
    './checkpoint',
    'Path for checkpoint directory.',
)


def main(argv):
    """
    Trains R2D2 agent on classic control tasks.
    """
    del argv

    runtime_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    logging.info(f'Run R2D2 agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    random_state = np.random.RandomState(FLAGS.seed)    

    # Create environment.
    def environment_builder():
        return gym_env.create_classic_environment(
            env_name=FLAGS.environment_name,
            seed=random_state.randint(1, 2**10),
        )

    eval_env = environment_builder()

    state_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.n

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Create network for learner to optimize, 
    # actor will use the same network with share memory.
    network = R2d2DqnMlpNet(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=FLAGS.learning_rate,
        eps=FLAGS.adam_eps,
    )

    # Test network output.
    obs = eval_env.reset()
    x = RnnDqnNetworkInputs(
        s_t=torch.from_numpy(obs[None, None, ...]).float(),
        a_tm1=torch.zeros(1, 1).long(),
        r_t=torch.zeros(1, 1).float(),
        hidden_s=network.get_initial_hidden_state(1),
    )
    network_output = network(x)
    assert network_output.q_values.shape == (1, 1, action_dim)
    assert len(network_output.hidden_s) == 2

    # Create prioritized transition replay, no importance_sampling_exponent decay
    importance_sampling_exponent = FLAGS.importance_sampling_exponent

    def importance_sampling_exponent_schedule(x):
        return importance_sampling_exponent
    
    replay = replay_lib.PrioritizedReplay(
        capacity=FLAGS.replay_capacity,
        structure=agent.TransitionStructure,
        priority_exponent=FLAGS.priority_exponent,
        importance_sampling_exponent=importance_sampling_exponent,
        normalize_weights=FLAGS.normalize_weights,
        random_state=random_state,
        time_major=True,
    )

    # Create queue to shared transitions between actors and learner
    data_queue = multiprocessing.Queue(maxsize=FLAGS.num_actors * 2)

    # Create shared objects so all actors processes can access them
    manager = multiprocessing.Manager()

    # Store copy of latest parameters of the neural network in a shared dictionary,
    # so actors can later access it
    shared_params = manager.dict({'network': None})

    # Create R2D2 learner instance
    learner_agent = agent.Leaner(
        network=network,
        optimizer=optimizer,
        replay=replay,
        min_replay_size=FLAGS.min_replay_size,
        target_net_update_interval=FLAGS.target_net_update_interval,
        discount=FLAGS.discout,
        burn_in=FLAGS.burn_in,
        priority_eta=FLAGS.priority_eta,
        rescale_epsilon=FLAGS.rescale_epsilon,
        batch_size=FLAGS.batch_size,
        n_step=FLAGS.n_step,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
        shared_params=shared_params,
    )

    # Create actor environments, actor instances.
    actor_envs = [
        environment_builder() for _ in range(FLAGS.num_actors)
    ]

    actor_devices = ['cpu'] * FLAGS.num_actors
    # Evenly distribute the actors to all available GPUs
    if torch.cuda.is_available() and FLAGS.actors_on_gpu:
        num_gpus = torch.cuda.device_count()
        actor_devices = [torch.device(
            f'cuda:{i % num_gpus}') for i in range(FLAGS.num_actors)]

    # Rank 0 is the most explorative actor, 
    # while rank N-1 is the most exploitative actor.
    # Each actor has it's own network with different weights.
    actors = [
        agent.Actor(
            rank=i,
            data_queue=data_queue,
            network=copy.deepcopy(network),
            random_state=np.random.RandomState(FLAGS.seed + int(i)),
            num_actors=FLAGS.num_actors,
            action_dim=action_dim,
            unroll_length=FLAGS.unroll_length,
            burn_in=FLAGS.burn_in,
            actor_update_interval=FLAGS.actor_update_ineterval,
            device=actor_devices[i],
            shared_params=shared_params,
        )
        for i in range(FLAGS.num_actors)
    ]

    # Create evaluation agent instance
    eval_agent = greedy_actors.R2d2EpsilonGreedActor(
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        random_state=random_state,
        device=runtime_device,
    )

    # Setup checkpoint.
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name,
        agent_name='R2D2',
        save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('network', network))

    # Run parallel training N iterations.
    main_loop.run_parallel_training_iterations(
        unm_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        learner_agent=learner_agent,
        eval_agent=eval_agent,
        eval_env=eval_env,
        actors=actors,
        actor_env=actor_envs,
        data_queue=data_queue,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    # Set multiprocessing start mode
    multiprocessing.set_start_method('spawn')
    app.run(main)