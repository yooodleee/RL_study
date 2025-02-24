
"""Helper functions for multi head/network (Ensemble-DQN and REM) agents."""

import collections
import typing
import numpy as np
import keras
import tensorflow as tf   # compat to version 1


MultiHeadNetworkType = collections.namedtuple(
    'multi_head_dqn_network', ['q_heads', 'unordered_q_heads', 'q_values']
)
DQNNetworkType = collections.namedtuple('dqn_network', ['q_values'])
MultiNetworkNetworkType = collections.namedtuple(
    'multi_network_dqn_network', ['q_networks', 'unordered_q_networks', 'q_values']
)
QuantileNetworkType = collections.namedtuple(
    'qr_dqn_network', ['q_values', 'logits', 'probabilities']
)



class QuantileNetwork(keras.Model):
    """Keras network for QR-DQN agent.
    
    
    Attributes
    ------------------
        num_actions: An integer representing the num of acts.
        num_atoms: An integer representing the num of quantiles of the value function
            distribution.
        conv1: First convolutional tf.keras layer with ReLU.
        conv2: Second convolutional tf.keras layer with ReLU.
        conv3: Third convolutional tf.keras layer with ReLU.
        flatten: A tf.keras Flatten layer.
        dense1: Penultimate fully-connected layer with ReLU.
        dense2: Final fully-connected layer with `num_actions` * `num_atoms` units.
    """

    def __init__(
            self,
            num_actions: int,
            num_atoms: int,
            name: str = 'quantile_network',
    ):
        """Convolutional network used to compute the agent's Q-value distribution.
        
        
        Args
        ------------------
            num_actions: (int) num of acts.
            num_atoms: (int) the num of buckets of the value function distribution.
            name: (str) used to create scope for network parameters.
        """
        super(QuantileNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        activation_fn = keras.activations.relu  # ReLU activation.
        self._kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )

        # Defining layers.
        self.conv1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=[8, 8],
            strides=4,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.conv2 = keras.layers.Conv2D(
            filters=64,
            kernel_size=[4, 4],
            strides=2,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.conv3 = keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            units=512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
        )
        self.dense2 = keras.layers.Dense(
            units=num_actions * num_atoms,
            kernel_initializer=self._kernel_initializer,
            activation=None,
        )
    

    def call(self, state):
        """Calculates the distribution of Q-values using the input state tensor."""
        net = tf.cast(state, tf.float32)
        net = tf.divide(net, 255.)
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.flatten(net)
        net = self.dense1(net)
        net = self.dense2(net)

        logits = tf.reshape(net, [-1, self.num_actions, self.num_atoms])
        probabilities = keras.activations.softmax(tf.zeros_like(logits))
        q_values = tf.reduce_mean(logits, axis=2)

        return QuantileNetworkType(q_values, logits, probabilities)



