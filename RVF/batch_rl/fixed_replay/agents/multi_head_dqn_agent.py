
"""
Multi Head DQN agent with fixed replay buffer(s).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import multi_head_dqn_agent


import gin
import tensorflow as tf # compat to version 1.x



@gin.configurable
class FixedReplayMultiHeadDQNAgent(multi_head_dqn_agent.MultiHeadDQNAgent):
    """
    MultiHeadDQNAgent with fixed replay buffer(s).
    """

    def __init__(
            self,
            sess,
            num_actions,
            replay_data_dir,
            replay_suffix=None,
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
                log Dir from which to load the replay buffer.
            replay_suffix: (int)
                If not None, then only load the replay buffer corresponding
                to the specific suffix in data dir.
            **kwargs:
                Arbtrary keyword args.
        """

        assert replay_data_dir is not None

        tf.compat.v1.logging.info(
            'Creating FixedReplayMultiHeadDQNAgent with replay directory: %s',
            replay_data_dir,
        )
        tf.compat.v1.logging.info(
            '\t replay_suffix %s',
            replay_suffix,
        )

        # set replay_log_dir before calling parent's initializer
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix

        super(
            FixedReplayMultiHeadDQNAgent, self
        ).__init__(
            sess, num_actions, **kwargs
        )


    def step(self, reward, observation):
        """
        Records the most recent transition and returns the agent's
            next action.


        Args:
        -------------
            reward: (float)
                the reward received from the agent's most recent action.
            observation: (np.array)
                the most recent observation.


        Returns: (int)
        -------------
            the selected action.
        """

        self._record_observation(observation)
        self.action = self._select_action()

        return self.action
    

    def end_episode(self, reward):
        assert self.eval_mode, 'Eval mode is not set to be True.'

        super(
            FixedReplayMultiHeadDQNAgent, self
        ).end_episode(reward)


    def _build_replay_buffer(self, use_staging):
        """
        Creates the replay buffer used by the agent.
        """

        return fixed_replay_buffer.WrappedFixedReplayBuffer(
            data_dir=self._replay_data_dir,
            replay_suffix=self._replay_suffix,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype
        )