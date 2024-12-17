from __future__ import print_function
import numpy as np
import tensorflow as tf
from constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer()

def cosineLoss(A, B, name):
    '''A, B : (BatchSize, d) '''
    dotprob = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A,1), tf.nn.l2_normalize(B,1)), 1)
    loss = 1 - tf.reduce_mean(dotprob, name=name)
    return loss

def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

def conv2d(
    x,
    num_filters,
    name,
    filter_size=(3, 3),
    stride=(1, 1),
    pad="SAME",
    dtype=tf.float32,
    collections=None
):
    with tf.compat.v1.get_variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int[x.get_shape()[3]], num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.compat.v1.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound), collections=collections)
        b = tf.compat.v1.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0), collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b

def deconv2d(
    x,
    out_shape,
    name,
    filter_size=(3, 3),
    stride=(1, 1),
    pad="SAME",
    dtype=tf.float32,
    collections=None,
    prevNumFeat=None,
):
    with tf.compat.v1.variable_scope(name):
        num_filters = out_shape[-1]
        prevNumFeat = int(x.get_shape()[3]) if prevNumFeat is None else prevNumFeat
        stride_shape = [1, stride[0], stride[1], 1]
        # transpose_filter : (height, width, out_channels, in_channels)
        filter_shape = [filter_size[0], filter_size[1], num_filters, prevNumFeat]

        # there are "num feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:2]) * prevNumFeat
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = np.prod(filter_shape[:3])
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.compat.v1.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound), collections=collections)
        b = tf.compat.v1.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0), collections=collections)
        # deconv2d = tf.reshape(tf.nn.bias_add(deconv2d, b), deconv2d.get_shape())
        return deconv2d
    
def linear()