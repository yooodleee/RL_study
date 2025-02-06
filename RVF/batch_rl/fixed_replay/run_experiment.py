
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
    

    