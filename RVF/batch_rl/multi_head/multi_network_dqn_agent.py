
"""Multi Q-Network DQN agent."""

import copy
import os
import gin
import tensorflow as tf


from batch_rl.multi_head import atari_helpers
from dopamine.agents.dqn import dqn_agent




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
        tf.compat.v1.logging.info('Creating MultiNetworkDQNAgent with following parameters:')
        tf.compat.v1.logging.info('\t num_networks: %d', num_networks)
        tf.compat.v1.logging.info('\t transform_strategy: %s', transform_strategy)
        tf.compat.v1.logging.info('\t num_convex_combinations: %d', num_convex_combinations)
        tf.compat.v1.logging.info('\t init_checkpoint_dir: %s', init_checkpoint_dir)
        tf.compat.v1.logging.info('\t use_deep_exploration: %s', use_deep_exploration)

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
        """Builds a multi-network Q-network that outputs Q-values for each network.
        
        
        Args
        ----------------
            name: (str) this name is passed to the tf.keras.Model and used to create
                variable scope under the hood by the tf.keras.Model.
                
                
        Returns
        ---------------
            network: (tf.keras.Model) the network instantiated by the Keras model.
        """
        
        # Pass the device_fn to place Q-networks on different devices
        kwargs = {'device_fn': lambda i: '/gpu: {}'.format(i // 4)}
        if self._q_networks_transform is None:
            if self.transform_strategy == 'STOCHASTIC':
                tf.compat.v1.logging.info('Creating q_networks transformation matrix..')
                self._q_networks_transform = atari_helpers.random_stochastic_matrix(
                    self.num_networks, num_cols=self._num_convex_combinations
                )
        
        if self._q_networks_transform is not None:
            kwargs.update({'transform_matrix': self._q_networks_transform})
        
        return self.network(
            num_actions=self.num_actions,
            num_networks=self.num_networks,
            transform_strategy=self.transform_strategy,
            name=name,
            **kwargs,
        )
    

    def _build_target_q_op(self):
        """Build an op used as a target for the Q-value.
        
        
        Returns
        ---------------
            target_q_op: An op calculating the Q-value.
        """

        # Get the maximum Q-value across the acts dims for each head.
        replay_next_qt_max = tf.reduce_max(
            self._replay_next_target_net_outputs.q_networks,
            axis=1,
        )
        is_non_terminal = 1. - tf.cast(self._replay.terminals, tf.float32)
        is_non_terminal = tf.expand_dims(is_non_terminal, axis=-1)
        rewards = tf.expand_dims(self._replay.rewards, axis=-1)

        return rewards + (
            self.cumulative_gamma * replay_next_qt_max * is_non_terminal
        )
    

    def begin_episode(self, observation):
        """Returns the agent's first act for this episode.
        
        
        Args
        ---------------
            observation: (np.array) the environment's initial obs.
            
            
        Returns
        ----------------
            (int) the selected act.
        """
        if self.use_deep_exploration:
            # Randomly pick a Q-function from all possible Q-function for data
            # collection each episode for online experiments, silimar to deep
            # exploration strategy proposed by Bootstrapped DQN
            self._sess.run(self._update_episode_q_function)
        
        return super(MultiNetworkDQNAgent, self).begin_episode(observation)
    

    def _build_networks(self):
        super(MultiNetworkDQNAgent, self)._build_networks()
        
        # q_argmax is only used for picking an act
        self._q_argmax_eval = tf.argmax(self._net_outputs.q_values, axis=1)[0]
        if self.use_deep_exploration:
            if self.transform_strategy.endswith('STOCHASTIC'):
                q_transform = atari_helpers.random_stochastic_matrix(
                    self.num_networks, num_cols=1
                )
                self._q_episode_transform = tf.compat.v1.get_variable(
                    trainable=False,
                    dtype=tf.float32,
                    shape=q_transform.get_shape().as_list(),
                    name='q_episode_transform',
                )
                self._update_episode_q_function = self._q_episode_transform.assign(
                    q_transform
                )
                episode_q_function = tf.tensordot(
                    self._net_outputs.unordered_q_networks,
                    self._q_episode_transform,
                    axes=[[2], [0]],
                )
                self._q_argmax_train = tf.argmax(episode_q_function[:, :, 0], axis=1)[0]
            
            elif self.transform_strategy == 'IDENTITY':
                self._q_function_index = tf.Variable(
                    initial_value=0,
                    trainable=False,
                    dtype=tf.int32,
                    shape=(),
                    name='q_head_episode',
                )
                self._update_episode_q_function = self._q_function_index.assign(
                    tf.random.uniform(
                        shape=(),
                        maxval=self.num_networks,
                        dtype=tf.int32,
                    )
                )
                q_function = self._net_outputs.unordered_q_networks[:, :, self._q_function_index]

                # This is only used for pickng an act
                self._q_argmax_train = tf.argmax(q_function, axis=1)[0]
            else:
                self._q_argmax_train = self._q_argmax_eval
    

    def _build_train_op(self):
        """Build a training op.
        
        
        Returns
        ---------------
            train_op: An op performing one step of training from replay data.
        """
        actions = self._replay.actions
        indices = tf.stack([tf.range(actions.shape[0]), actions], axis=-1)
        replay_chosen_q = tf.gather_nd(
            self._replay_net_outputs.q_networks,
            indices=indices,
        )
        target = tf.stop_gradient(self._build_target_q_op())
        loss = tf.compat.v1.losses.huber_loss(
            target,
            replay_chosen_q,
            reduction=tf.compat.v1.losses.Reduction.NONE,
        )
        q_head_losses = tf.reduce_mean(loss, axis=0)
        final_loss = tf.reduce_mean(q_head_losses)

        if self.summary_writer is not None:
            with tf.compat.v1.variable_scope('Losses'):
                tf.summary.scalar('HuberLoss', final_loss)
        
        self.optimizers = [copy.deepcopy(self.optimizers) for _ in range(self.num_networks)]

        train_ops = []
        for i in range(self.num_networks):
            var_list = tf.compat.v1.trainable_variables(scope='Online/subnet_{}'.format(i))
            train_op = self.optimizers[i].minimize(final_loss, var_list=var_list)
            train_ops.append(train_op)
        
        return tf.group(*train_ops, name='marged_train_op')