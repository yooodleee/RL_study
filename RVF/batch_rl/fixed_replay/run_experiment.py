
"""
Runner for experiments with a fixed replay buffer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from dopamine.discrete_domains import checkpointer
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment


import gin
import tensorflow as tf # compat to version 1.x




@gin.configurable
class FixedReplayRunner(run_experiment.Runner):
    """
    Object that handles running Dopamine experiments with fixed replay
        buffer.
    """

    def _initialize_checkpointer_and_maybe_resume(
            self, 
            checkpoint_file_prefix):
        
        super(FixedReplayRunner, self)._initialize_checkpointer_and_maybe_resume(
            checkpoint_file_prefix
        )


        # code for the loading a checkpoint at initialization
        init_checkpoint_dir = self._agent._init_checkpoint_dir

        if (self._start_iteration == 0) and (init_checkpoint_dir is not None):

            if checkpointer.get_latest_checkpoint_number(
                self._checkpoint_dir
            ) < 0:
                
                # No checkpoint loaded yet, read init_checkpoint_dir
                init_checkpointer = checkpointer.Checkpointer(
                    init_checkpoint_dir,
                    checkpoint_file_prefix,
                )
                latest_init_checkpoint = checkpointer.get_latest_checkpoint_number(
                    init_checkpoint_dir
                )

                if latest_init_checkpoint >= 0:
                    experimenta_data = init_checkpointer.load_checkpoint(
                        latest_init_checkpoint
                    )
                    if self._agent.unbundle(
                        init_checkpoint_dir,
                        latest_init_checkpoint,
                        experimenta_data,
                    ):
                        if experimenta_data is not None:
                            assert 'logs' in experimenta_data
                            assert 'current_iteration' in experimenta_data

                            self._logger.data = experimenta_data['logs']
                            self._start_iteration = experimenta_data['current_iteration'] + 1
                        
                        tf.compat.v1.logging.info(
                            'Reloaded checkpoint from %s and will start iteration %d',
                            init_checkpoint_dir,
                            self._start_iteration,
                        )
    

    def _run_train_phase(self):
        """
        Run training phase.
        """

        self._agent.eval_mode = False
        start_time = time.time()

        for _ in range(self._training_steps):
            self._agent._train_step()
        
        time_delta = time.time() - start_time

        tf.compat.v1.logging.info(
            'Average training steps per second: %.2f',
            self._training_steps / time_delta,
        )
    


    def _run_one_iteration(self, iteration):
        """
        Runs one iter of agent/env iteraction.
        """

        statistics = iteration_statistics.IterationStatistics()

        tf.compat.v1.logging.info(
            'Starting iteration %d',
            iteration,
        )

        if not self._agent._replay_suffix:
            # reload the replay buffer
            self._agent._replay.memory.reload_buffer(
                num_buffers=5
            )
        
        self._run_train_phase()

        
        num_episode_eval, \
        average_reward_eval = self._run_eval_phase(
            statistics
        )

        self._save_tensorboard_summaries(
            iteration,
            num_episode_eval,
            average_reward_eval,
        )

        return statistics.data_lists
    


    