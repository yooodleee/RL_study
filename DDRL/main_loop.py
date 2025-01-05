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


def run_single_thread_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    train_agent: types_lib.Agent,
    train_env: gym.Env,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
)-> None:
    """
    Runs single-thread training and evaluation for N iterations.
    The same code structure is shared by most single-threaded DQN agents,
        and some policy gradients agents like reinforce, actor-critic.

    For every iteration:
        1. Start to run agent for num_train_steps training environment steps/frames.
        2. Create checkpoint file.
        3. (Optional) Run some evaluation steps with a seperate evaluation actor and
            environment.
    
    Args:
        num_iterations: number of iterations to run.
        num_train_steps: number of frames (or env steps) to run, per iteration.
        num_eval_steps: number of evaluation frames (or env steps) to run, per iteration.
        train_agent: training agent, expect the agent to have step(), reset(), and a 
            agent_name property.
        train_env: training environment.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        use_tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag, default None.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard,
            default 0 no screenshots.
    """

    # Create log file writer.
    writer = CsvWriter(csv_file)

    # Create trackers for training and evaluation
    train_tb_log_prefix = (
        get_tb_log_prefix(
            train_env.spec.id,
            train_agent.agent_name,
            tag,
            'train',
        ) if use_tensorboard else None
    )
    train_trackers = trackers_lib.make_default_trackers(
        train_tb_log_prefix, debug_screenshots_interval
    )

    should_run_evaluator = False
    eval_trackers = None
    if num_eval_steps > 0 and eval_agent is not None \
        and eval_env is not None:
        eval_tb_log_prefix = (
            get_tb_log_prefix(
                eval_env.spec.id,
                eval_agent.agent_name,
                tag,
                'eval',
            ) if use_tensorboard else None
        )
        eval_trackers = trackers_lib.make_default_trackers(
            eval_tb_log_prefix, debug_screenshots_interval
        )
    
    # Start training
    for iteration in range(1, num_iterations + 1):
        logging.info(
            f'Training iteration {iteration}'
        )

        # Run training steps.
        train_stats = run_env_steps(
            num_train_steps,
            train_agent,
            train_env,
            train_trackers,
        )

        checkpoint.set_iteration(iteration)
        saved_ckpt = checkpoint.save()

        if saved_ckpt:
            logging.info(
                f'New checkpoint created at "{saved_ckpt}"'
            )
        
        # Logging training statistics.
        log_output = [
            ('iteration', iteration, '%3d'),
            ('train_step', iteration * num_train_steps, '%5d'),
            ('train_episode_return', train_stats['mea_episode_return'], '%2.2f'),
            ('train_num_episodes', train_stats['num_episodes'], '%3d'),
            ('train_step_rate', train_stats['step_rate'], '%4.0f'),
            ('train_duration', train_stats['duration'], '%.2f'),
        ]

        # Run evaluation steps.
        if should_run_evaluator is True:
            logging.info(
                f'Evaluataion iteration {iteration}'
            )

            # Run some evaluation steps.
            eval_stats = run_env_steps(
                num_eval_steps,
                eval_agent,
                eval_env,
                eval_trackers,
            )

            # Logging evaluation statistics.
            eval_output = [
                ('eval_step', iteration * num_eval_steps, '%5d'),
                ('eval_episode_return', eval_stats['mean_episode_return'], '%2.2f'),
                ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
                ('eval_step_rate', eval_stats['step_rate'], '%4.0f'),
                ('eval_duration', eval_stats['duration'], '%.2f'),
            ]
            log_output.extend(eval_output)
        
        log_output_str = ', '.join(
            ('%s: ' + f) % (n, v) for n, v, f in log_output
        )
        logging.info(log_output_str)
        writer.write(
            collections.OrderedDict(
                (n, v) for n, v, _ in log_output
            )
        )
    writer.close()


def run_parallel_training_iterations(
    num_iterations: int,
    num_train_steps: int,
    num_eval_steps: int,
    learner_agent: types_lib.Learner,
    eval_agent: types_lib.Agent,
    eval_env: gym.Env,
    actors: List[types_lib.Agent],
    actor_envs: List[gym.Env],
    data_queue: multiprocessing.Queue,
    checkpoint: PyTorchCheckpoint,
    csv_file: str,
    use_tensorboard: bool,
    tag: str = None,
    debug_screenshots_interval: int = 0,
)-> None:
    """
    This is the place to kick start parallel training with multiple actors processes
        and a single learner process.

    Args:
        num_iterations: number of iterations to run.
        num_train_steps: number of frames (or env steps) to run, per iteration.
        num_eval_steps: number of evaluation frames (or env steps) to run, per 
            iteration.
        learner_agent: learner agent, expect the agent to have run_train_loop() 
            method.
        eval_agent: evaluation agent.
        eval_env: evaluation environment.
        actors: list of actor instances we wish to run.
        actor_envs: list of gym.Env for each actor to run.
        data_queue: a multiprocessing.Queue used to recieve transition samples from actors.
        checkpoint: checkpoint object.
        csv_file: csv log file path and name.
        use_tensorboard: if True, use tensorboard to log the runs.
        tag: tensorboard run log tag, default None.
        debug_screenshots_interval: the frequency to take screenshots and add to tensorboard,
            default 0 no screenshots.
    """

    # Create shared iteration count and start, end trainng event.
    # start_iteration_event is used to signaling actors to run one training iteration,
    # stop_event is used to signaling actors the end of training session.
    # The start_iteration_event and stop_event are only set by the main process.
    iteration_count = multiprocessing.Value('i', 0)
    start_iteration_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    # To get training statistics from each actor and the learner.
    # Use a single writer to write to csv file.
    log_queue = multiprocessing.SimpleQueue()

    # Run learner train loop on a new thread.
    learner = threading.Thread(
        target=run_learner,
        args=(
            num_iterations,
            num_eval_steps,
            learner_agent,
            eval_agent,
            eval_env,
            data_queue,
            log_queue,
            iteration_count,
            start_iteration_event,
            stop_event,
            checkpoint,
            len(actors),
            use_tensorboard,
            tag,
        ),
    )
    learner.start()

    # Start logging on a new thread, since it's very light-weight task.
    logger = threading.Thread(
        target=run_logger,
        args=(log_queue, csv_file),
    )
    logger.start()

    # Create and start actor processes once, this will preserve actor's 
    # internal like steps etc.
    # Tensorboard log dir prefix.
    num_actors = len(actors)
    actor_tb_log_prefixes = [None for _ in range(num_actors)]
    if use_tensorboard:
        # To get better performance, only log a maximum of 8 actor statistics 
        # to tensorboard
        _step = 1 if num_actors <= 8 else math.ceil(num_actors / 8)
        for i in range(0, num_actors, _step):
            actor_tb_log_prefixes[i] = get_tb_log_prefix(
                actor_envs[i].spec.id,
                actors[i].agent_name,
                tag,
                'train',
            )
    
    processes = []
    for actor, actor_env, tb_log_prefix in zip(
        actors, actor_envs, actor_tb_log_prefixes
    ):
        p = multiprocessing.Process(
            target=run_actor,
            args=(
                actor,
                actor_env,
                data_queue,
                log_queue,
                num_train_steps,
                iteration_count,
                start_iteration_event,
                stop_event,
                tb_log_prefix,
                debug_screenshots_interval,
            ),
        )
        p.start()
        processes.append(p)
    
    # Wait for all actor to be finished.
    for p in processes:
        p.join()
        p.close()
    
    # learner.join()
    logger.join()

    # Close queue.
    data_queue.close()


