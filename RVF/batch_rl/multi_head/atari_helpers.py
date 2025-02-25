
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



class MultiHeadQNetwork(keras.Model):
    """Multi-head convolutional network to compute multiple Q-value estimates.
    
    
    Attributes
    -----------------------
        num_actions: An integer representing the num of acts.
        num_heads: An integer representing the num of Q-heads.
        conv1: First convolutional keras.layer with ReLU.
        conv2: Second convolutional keras.layer with ReLU.
        conv3: Third convolutional keras.layer with ReLU.
        flatten: A keras.Flatten layer.
        dense1: Penultimate fully-connected layer with ReLU.
        dense2: Final fully-connected layer with `num_actions` * `num_heads` units.
    """

    def __init__(
            self,
            num_actions: int,
            num_heads: int,
            transform_strategy: typing.Optional[str] = None,
            name: typing.Optional[str] = None,
            **kwargs,
    ):
        """Creates the layers used calculating return distributions.
        
        
        Args
        ------------------
            num_actions: num of acts.
            num_heads: num of Q-heads.
            transform_strategy: Possible options include (1) 'IDENTITY' for no
                transformation (Ensemble-DQN) (2) 'STOCHASTIC' for random convex
                combination (REM).
                
            name: used to create scope for network parameters.
            **kwargs: Arbitrary ketword args. Used for passing 'transform_matrix',
                the matrix transforming the Q-values if the passed 'transform_strategy'
                is 'STOCHASTIC'.
        """
        super(MultiHeadQNetwork, self).__init__(name=name)
        activation_fn = keras.activations.relu
        self.num_actions = num_actions
        self.num_heads = num_heads
        self._transform_strategy = transform_strategy
        self._kwargs = kwargs
        self._kernel_initializer = keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform'
        )

        # Defining layers.
        self.conv1 = keras.layers.Conv2D(
            32,
            [8, 8],
            strides=4,
            padding='same'
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
            name='Conv',
        )
        self.conv2 = keras.layers.Conv2D(
            64,
            [4, 4],
            strides=2,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
            name='Conv',
        )
        self.conv3 = keras.layers.Conv2D(
            64,
            [3, 3],
            strides=1,
            padding='same',
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
            name='Conv',
        )
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            512,
            activation=activation_fn,
            kernel_initializer=self._kernel_initializer,
            name='fully_connected',
        )
        self.dense2 = keras.layers.Dense(
            num_actions * num_heads,
            kernel_initializer=self._kernel_initializer,
            name='fully_connected',
        )
    

    def call(self, state):
        """Creates the output tensor/op given the input state tensor.
        
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        info on this. Note that keras.Model implements `call` which is
        wrapped by `__call__` function by keras.Model.
        
        
        Args
        ----------------
            state: (Tensor) input tensor.
            
            
        Returns
        -----------------
            (Collection.namedtuple) output ops (graph mode) or output tensor (eager).
        """
        net = tf.cast(state, tf.float32)
        net = tf.divide(net, 255.)
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.flatten(net)
        net = self.dense1(net)
        net = self.dense2(net)

        unordered_q_heads = tf.reshape(net, [-1, self.num_actions, self.num_heads])
        q_heads, q_values = combine_q_functions(
            unordered_q_heads,
            self._transform_strategy,
            **self._kwargs,
        )

        return MultiHeadNetworkType(q_heads, unordered_q_heads, q_values)
    


def combine_q_functions(
        q_functions,
        transform_strategy,
        **kwargs
):
    """Utility function for combining multiple Q functions.
    
    
    Args
    -----------------
        q_function: Multiple Q-funtion concatenated.
        transform_strategy: (str) Possible options include (1) 'IDENTITY' for no
            transform (2) 'STOCHASTIC' for random convex combination.
        **kwargs: Arbitrary ketword args. Used for passing `transform_matrix`,
            the matrix for transforming the Q-values if the passed 
            `transform_strategy` is `STOCHASTIC`.
    
    
    Returns
    ------------------
        q_functions: Modified Q-functions.
        q_values: Q-values based on combining the multiple heads.
    """

    # Create q_values before reordering the heads for training
    q_values = tf.reduce_mean(q_functions, axis=-1)

    if transform_strategy == 'STOCHASTIC':
        left_stochastic_matrix = kwargs.get('transform_matrix')
        if left_stochastic_matrix is None:
            raise ValueError('None value provided for stochastic matrix')
        
        q_functions = tf.tensordot(
            q_functions, left_stochastic_matrix, axes=[[2], [0]]
        )
    elif transform_strategy == 'IDENTITY':
        tf.logging.info('Identity transformation Q-function heads')
    else:
        raise ValueError(
            '{} is not a valid reordering strategy'.format(transform_strategy)
        )
    return q_functions, q_values



class NatureDQNNetwork(keras.Model):
    """The convolutional network used to compute the agent's Q-values.
    
    
    Attributes
    ----------------------
        num_actions: An integer representing the num of acts.
        conv1: First convolutional keras.layer with ReLU.
        conv2: Second convolutional keras.layer with ReLU.
        conv3: Third convolutional keras.layer with ReLU.
        flatten: A keras.Flatten layer.
        dense1: Penultimate fully-connected layer with ReLU.
        dense2: Final fully-connected layer with `num_actions` units.
    """

    def __init__(
            self,
            num_actions: int,
            name: typing.Optional[str] = None,
    ):
        """Creates the layer used for calculating Q-values.
        
        
        Args
        -------------
            num_actions: num of acts.
            name: used to create scope for network params.
        """
        super(NatureDQNNetwork, self).__init__(name=name)
        self.num_actions = num_actions
        
        # Defining layers.
        activation_fn = keras.activations.relu
        
        # setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = keras.layers.Conv2D(
            32,
            [8, 8],
            strides=4,
            padding='same',
            activation=activation_fn,
            name='Conv',
        )
        self.conv2 = keras.layers.Conv2D(
            64,
            [4, 4],
            strides=2,
            padding='same',
            activation=activation_fn,
            name='Conv',
        )
        self.conv3 = keras.layers.Conv2D(
            64,
            [3, 3],
            strides=1,
            padding='same',
            activation=activation_fn,
            name='Conv',
        )
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(
            512, activation=activation_fn, name='fully_connected'
        )
        self.dense2 = keras.layers.Dense(num_actions, name='fully_connected')
    

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.
        
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        info on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.
        
        Params created here will have scope according to the `name` argument
        given at `.__init__()` call.
        
        
        Args
        --------------
            state: (Tensor) input tensor.
            
            
        Returns
        --------------
            (collections.namedtuple) output ops (graph mode) or output tensor (eager).
        """
        net = tf.cast(state, tf.float32)
        net = tf.divide(net, 255.)
        net = self.conv1(net)
        net = self.conv2(net)
        net = self.conv3(net)
        net = self.flatten(net)
        net = self.dense1(net)

        return DQNNetworkType(self.dense2(net))



class MultiNetworkQNetwork(keras.Model):
    """Multiple convolutional network to compute Q-values estimates.
    
    
    Attributes
    ---------------------
        num_actions: An integer representing the num of acts.
        num_networks: An integer representing the num of Q-networks.
    """

    def __init__(
            self,
            num_actions: int,
            num_networks: int,
            transform_strategy: typing.Optional[str] = None,
            name: typing.Optional[str] = None,
            **kwargs,
    ):
        """Creates the networks used calculating multiple Q-values.
        
        
        Args
        ---------------
            num_actions: num of acts.
            num_networks: num of separate Q-networks.
            transform_strategy: Possible options include (1) 'IDENTITY' for no
                transformation (Ensemble-DQN) (2) 'STOCHASTIC' for random convex
                combination (REM).
            name: used to create scope for network params.
            **kwargs: Arbitrary ketword args. Used for passing 'transform_matrix',
                the matrix for transforming the Q-values if only the passed 
                `transform_strategy` is `STOCHASTIC`.
        """
        super(MultiNetworkQNetwork, self).__init__(name=name)
        
        self.num_actions = num_actions
        self.num_networks = num_networks
        self._transform_strategy = transform_strategy
        self._kwargs = kwargs
        self._device_fn = kwargs.pop('device_fn', lambda i: '/gpu:0')

        # Create multiple Q-networks
        self._q_networks = []
        for i in range(self.num_networks):
            with tf.device(self._device_fn(i)):
                q_net = NatureDQNNetwork(num_actions, name='subnet_{}'.format(i))
            self._q_networks.append(q_net)
    

    