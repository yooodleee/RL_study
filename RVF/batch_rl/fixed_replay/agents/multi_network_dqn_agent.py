
"""
Multi Network DQN agent with fixed replay buffer(s).
"""


from batch_rl.fixed_replay.replay_memory import fixed_replay_buffer
from batch_rl.multi_head import multi_network_dqn_agent


import gin
import tensorflow as tf     # compat to version 1.x



@gin.configurable
class FixedReplayMultiNetworkDQNAgent(
    multi_network_dqn_agent.MultiNetworkDQNAgent):

    """
    MultiNetworkDQNAgent with fixed replay buffer(s).
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
                log dir from which to load the replay buffer.
            replay_suffix: (int)
                If not None, then only load the replay buffer corresponding
                to the specific suffix in data dir.
            **kwargs:
                Arbtrary keyword args.
        """

        assert replay_data_dir is not None

        tf.compat.v1.logging.info(
            'Creating FixedReplayMultiNetworkDQNAgent with replay directory: %s',
            replay_data_dir
        )
        tf.compat.v1.logging.info(
            '\t replay_suffix %s',
            replay_suffix
        )

        # set replay_log_dir before calling parent's initializer
        self._replay_data_dir = replay_data_dir
        self._replay_suffix = replay_suffix

        super(
            FixedReplayMultiNetworkDQNAgent, self
        ).__init__(
            sess, num_actions, **kwargs
        )


    def step(self, reward, observation):
        """
        Records the most recent transition and returns the agent's next action.


        Args:
        --------------
            reward: (float)
                the reward received from the agent's most recent action.
            observation: (np.array)
                the most recent observation.


        Returns: (int)
        ----------------
            the selected action.
        """

        self._record_observation(observation)
        self.action = self._select_action()

        return self.action
    

    