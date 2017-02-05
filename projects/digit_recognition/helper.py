import tensorflow as tf


def var(kernel_shape):
    """weights biases var init"""
    weight = tf.get_variable("weights", kernel_shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("biases", [kernel_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    return weight, biases


# Conv Layer Model
def conv_relu(x_input, kernel_shape, pool=False, drop=None):
    """build conv relu layer"""
    weights, biases = var(kernel_shape)
    conv = tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='SAME')
    rtn = tf.nn.relu(conv + biases)
    if pool:
        rtn = tf.nn.max_pool(rtn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    return rtn


def relu(x_input, kernel_shape, drop=None):
    """build relu layer"""
    weights, biases = var(kernel_shape)
    rtn = tf.nn.relu(tf.matmul(x_input, weights) + biases)
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    return rtn

