
"""
Quantile Regression agent (QR-DQN) with fixed replay buffer(s).
"""


from __future__ import absolute_import
from __future__ import division 
from __future__ import print_function


import os


from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import quantile_agent


import gin
import tensorflow as tf     # compat to version 1.x



@gin.configurable
class FixedReplayQuantileAgent(quantile_agent.QuantileAgent):
    """
    An implementation of the DQN agent with fixed replay buffer(s).
    """

    def __init__(
            self,
            sess,
            num_actions,
            replay_data_dir,
            replay_suffix=None,
            init_checkpoint_dir=None,
            **kwargs):
        
        """
        Initializes the agent and constructs the components of its graph.


        Args:
        -------------
            sess: (tf.Session)
                for executing ops.
            num_actions: (int)
                number of acts the agent can take at any state.
            replay_data_dir: (str)
                log dir from which to load the replay buffer.
            replay_suffix: (int)
                If not None, then only load the replay buffer corresponding
                the replay buffer.
            init_checkpoint_dir: (str)
                dir from which initial checkpoint before training is loaded 
                if there doesn't exist any checkpoint in the current agent
                dir. If None, no initial checkpoint is loaded.
            **kwargs:
                Arbtrary keyword args.
        """

        assert replay_data_dir is not None

        # set replay_log_dir before calling parent's initializer
        tf.compat.v1.logging.info(
            'Creating FixedReplayAgent with replay directory: %s',
            replay_data_dir
        )
        tf.compat.v1.logging.info(
            '\t init_checkpoint_dir: %s',
            init_checkpoint_dir
        )
        tf.compat.v1.logging.info(
            '\t replay_suffix %s',
            replay_suffix
        )

        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix


        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir,
                'checkpoints'
            )
        else:
            self._init_checkpoint_dir = None
        
        super(
            FixedreplayQuantileAgent, self
        ).__init__(
            sess, num_actions, **kwargs
        )
    

