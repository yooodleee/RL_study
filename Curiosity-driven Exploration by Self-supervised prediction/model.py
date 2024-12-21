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
    
def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.compat.v1.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.compat.v1.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(w, x) + b

def categorical_sample(logits, d):
    value = tf.squeeze(tf.compat.v1.multinomial(logits - tf.reduce_max(logits, [1], keepdims=True), 1), [1])
    return tf.one_hot(value, d)

def inverseUniverseHead(x, final_shape, nConvs=4):
    """
        universe agent example
        input: [None, 288]; output: [None, 42, 42, 1]:
    """
    print('Using inverse-universe head design')
    bs = tf.shape(x)[0]
    deconv_shape1 = [final_shape[1]]
    deconv_shape2 = [final_shape[2]]
    for i in range(nConvs):
        deconv_shape1.append((deconv_shape1[-1]-1)/2 + 1)
        deconv_shape2.append((deconv_shape2[-1]-1)/2 + 1)
    inshapeprod = np.prod(x.get_shape().as_list()[1:]) / 32.0
    assert(inshapeprod == deconv_shape1[-1] * deconv_shape2[-1])
    # print('deconv_shape1: ', deconv_shape1)
    # print('deconv_shape2: ', deconv_shape2)

    x = tf.reshape(x, [-1, deconv_shape1[-1], deconv_shape2[-1], 32])
    deconv_shape1 = deconv_shape1[:-1]
    deconv_shape2 = deconv_shape2[:-1]
    for i in range(nConvs - 1):
        x = tf.nn.elu(deconv2d(x, [bs, deconv_shape1[-1], deconv_shape2[-1], 32],
                            "dl{}".format(i + 1), [3, 3], [2, 2], prevNumFeat=32))
        deconv_shape1 = deconv_shape1[:-1]
        deconv_shape2 = deconv_shape2[:-1]
    x = deconv2d(x, [bs] + final_shape[1:], "dl4", [3, 3], [2, 2], prevNumFeat=32)
    return x


def universeHead(x, nConvs=4):
    """
        uinverse agent example
        input" [None, 42, 42, 1]; output: [None, 288];
    """
    print('Using universe head design')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # print('Loop{} '.format(i+1), tf.shape(x))
        # print('Loop{} '.format(i+1), x.get_shape())
    x = flatten(x)
    return x

def nipHead(x):
    """
        DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    """
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x

def natureHead(x):
    """
        DQN Nature 2015 paper
        input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
    """
    print('Using nature head design')
    x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 512, "fc", normalized_columns_initializer(0.01)))
    return x


def doomHead(x):
    """
        Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
    """
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x