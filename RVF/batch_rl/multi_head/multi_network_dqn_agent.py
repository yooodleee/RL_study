
"""Multi Q-Network DQN agent."""

import copy
import os
import gin
import tensorflow as tf


from batch_rl.multi_head import atari_helpers
import dopamine     # dopamine.agents.dqn => dqn_agent




@gin.configurable
class MultiNetworkDQNAgent(dqn_agent.DQNAgent):
    """DQN agent with multiple heads."""

    def __init__(
            self,
            sess,
            num_actions,
            num_networks=1,
            transform_strategy='IDENTITY',
            num_convex_combinations=1,
            network=atari_helpers.MultiNetworkQNetwork,
            init_checkpoint_dir=None,
            use_deep_exploration=False,
            **kwargs,
    ):
        """Initializes the agent and constructs the components of its graph.
        
        
        Args
        ----------------
            sess: (tf.Session) for executing ops.
            num_actions: (int) num of acts the agent can take at any state.
            num_networks: (int) num of different Q-functions.
            transform_strategy: (str) Possible options include (1) 'STOCHASTIC' for
                multiplication with a left stochastic matrix. (2) 'IDENTITY', in which
                case the heads are not transformed.
            num_convex_combinations: If transform_strategy is 'STOCHASTIC',
                then this arg specifies the num of random convex combinations
                to be created. If None, `num_heads` convex combinations are created.
            network: (tf.keras.Model) A call to this obj will return an instantiation
                of the network provided. The network returned can be run with different
                inputs to create different outputs. See atari_helpers.MultiNetworkQNetowkr
                as an example.
            init_checkpoint_dir: (str) directory from which initial checkpoint before
                training is loaded if there doesn't exist any checkpoint in the current
                agent dir. If None, no initial checkpoint is loaded.
            use_deep_exploration: Adaptation of Bootstrapped DQN for REM exploration.
            **kwargs: Arbitrary ketwor args.
        """
        tf.logging.info('Creating MultiNetworkDQNAgent with following parameters:')
        tf.logging.info('\t num_networks: %d', num_networks)
        tf.logging.info('\t transform_strategy: %s', transform_strategy)
        tf.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
        tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        tf.logging.info('\t use_deep_exploration: %s', use_deep_exploration)

        self.num_networks = num_networks
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints'
            )
        else:
            self._init_checkpoint_dir = None
        
        # The transform matrix should be created on device specified by tf_device
        # if the transform_strategy is UNIFORM_STOCHASTIC or STOCHASTIC
        self._q_networks_transform = None
        self._num_convex_combinations = num_convex_combinations
        self.transform_strategy = transform_strategy
        self.use_deep_exploration = use_deep_exploration
        
        super(MultiNetworkDQNAgent, self).__init__(
            sess, num_actions, network=network, **kwargs
        )
    

    def _create_network(self, name):
        