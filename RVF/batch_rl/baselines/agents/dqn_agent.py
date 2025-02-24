
"""
DQN Agent with logged replay buffer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from batch_rl.baselines.replay_memory import logged_replay_buffer
from dopamine.agents.dqn import dqn_agent

import gin


@gin.configurable
class LoggedDQNAgent(dqn_agent.DQNAgent):
    """
    An implementation of the DQN agent with replay buffer logging to disk.
    """

    def __init__(
            self,
            sess,
            num_actions,
            replay_log_dir,
            **kwargs):
        
        """
        Initializes the agent and constructs the components of its graph.

        Args:
            sess(tr.Session): for executing ops.
            num_actions(int): number of actions the agent can take at any state.
            replay_log_dir(str): log Directory to save the replay buffer to disk
                periodically.
            **kwargs: Arbitrary keyword arguments.
        """
        assert replay_log_dir is not None
        # Set replay_log_dir before calling parent's initializer
        self._replay_log_dir = replay_log_dir
        super(LoggedDQNAgent, self).__init__(sess, num_actions, **kwargs)
    
    def log_final_buffer(self):
        self._replay.memory.log_final_buffer()
    
    def _build_replay_buffer(self, use_staging):
        """
        Creates the replay buffer used by the agent.

        Args:
            use_staging(bool): if True, uses a staging area to prefetch data for
                faster training.

        Returns:
            A WrapperReplayBuffer object.
        """
        return logged_replay_buffer.WrappedLoggedReplayBuffer(
            log_dir=self._replay_log_dir,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype,
        )