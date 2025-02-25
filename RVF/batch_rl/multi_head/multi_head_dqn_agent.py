
"""Multi Head DQN agent."""

import os
import gin
import tensorflow as tf


from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent



@gin.configurable
class MultiHeadDQNAgent(dqn_agent.DQNAgent):
    """DQN agent with multiple heads."""

    def __init__(
            self,
            sess,
            num_actions,
            num_heads=1,
            transform_strategy='IDENTITY',
            num_convex_combinations=1,
            network=atari_helpers.MultiHeadQNetwork,
            init_checkpoint_dir=None,
            **kwargs,
    ):
        """Initializes the agent constructs the components of its graph.
        
        
        Args
        -----------------
            sess: (tf.Session) for executing ops.
            num_actions: (int) num of acts the agent can take at any state.
            num_heads: (int) num of heads per act output of the Q function.
            transform_strategy: (str) Possible options include (1)
                'STOCHASTIC' for multiplication with a left stochastic matrix.
                (2) 'IDENTITY', ,in which case the heads are not transformed.
            num_convex_combinations: If transform_strategy is 'STOCHASTIC',
                then this arg specifies the num of random convex combinations 
                to be created. If None, `num_heads` convex combinations are created.
            network: (tf.keras.Model) A call to this obj will return an instantiation
                of network provided. The network returned can be run with different
                inputs to create different outputs. See atari_helpers.MultiHeadQNetwork
                as an example.
            init_checkpoint_dir: (str) dir from which initial checkpoint before training
                in the current agent dir. If None, no initial checkpoint is loaded.
            **kwargs: Arbitrary ketwor dargs.
        """
        tf.logging.info('Creating MultiHeadDQNAgent with following parameters:')
        tf.logging.info('\t num_heads: %d', num_heads)
        tf.logging.info('\t transform_strategy: %s', transform_strategy)
        tf.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
        tf.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        
        self.num_heads = num_heads
        if init_checkpoint_dir is not None:
            self._init_checkpoint_dir = os.path.join(
                init_checkpoint_dir, 'checkpoints'
            )
        else:
            self._init_checkpoint_dir = None
        
        self._q_heads_transform = None
        self._num_convex_combinations = num_convex_combinations
        self.transform_strategy = transform_strategy
        super(MultiHeadDQNAgent, self).__init__(
            sess, num_actions, network=network, **kwargs
        )

    
    def _create_network(self, name):
        """Builds a multi-head Q-network that outputs Q-values for multiple heads.
        
        
        Args
        ---------------
            name: (str) this name is passed to the tf.keras.Model and used to create
                variable scope under the hood by the tf.keras.Model.
        
        
        Returns
        -----------------
            network: (tf.keras.Model) the network instantiated by the keras model.
        """
        kwargs = {} # Used for passing the transformation matrix if any
        if self._q_heads_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.logging.info('Creating q_heads transformation matrix..')
                self._q_heads_transform = atari_helpers.random_stochastic_matrix(
                    self.num_heads,
                    num_cols=self._num_convex_combinations,
                )
        if self._q_heads_transform is not None:
            kwargs.update({'transform_matrix': self._q_heads_transform})
        
        network = self.network(
            num_actions=self.num_actions,
            num_heads=self.num_heads,
            transform_strategy=self.transform_strategy,
            name=name,
            **kwargs,
        )

        return network
    

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-values.
        
        
        Returns
        --------------  
            target_q_op: An op calculating the Q-value.
        """

        # Get the maximum Q-value across the acts dim for each head.
        replay_next_qt_max = tf.reduce_max(
            self._replay_next_target_net_outputs.q_heads, axis=1
        )
        is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
        is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
        rewards = tf.expand_dims(self._replay.rewards, axis=-1)

        return rewards + (
            self.cumulative_gamma * replay_next_qt_max * is_non_terminal
        )
    

    def _build_train_op(self):
        """Builds a training op.
        
        
        Returns
        -------------
            train_op: An op performing one step of training from replay data.
        """
        actions = self._replay.actions
        indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
        replay_chosen_q = tf.gather_nd(
            self._replay_net_outputs.q_heads, indices=indices
        )
        target = tf.stop_gradient(self._build_target_q_op())
        loss = tf.losses.hubeer_loss(
            target,
            replay_chosen_q,
            reduction=tf.losses.Reduction.NONE,
        )
        q_heads_losses = tf.reduce_mean(loss, axis=0)
        final_loss = tf.reduce_mean(q_heads_losses)

        if self.summary_writer is not None:
            with tf.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', final_loss)
        
        return self.optimizer.minimize(final_loss)