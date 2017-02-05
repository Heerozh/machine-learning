import numpy as np
import tensorflow as tf
import functools

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
    print(rtn)
    return rtn


def relu(x_input, kernel_shape, drop=None):
    """build relu layer"""
    weights, biases = var(kernel_shape)
    rtn = tf.nn.relu(tf.matmul(x_input, weights) + biases)
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    print(rtn)
    return rtn


class Learner:
    batch_size = 128
    func_model = None
    func_accuracy = None
    steps = 1001
    func_loss = None
    func_optimizer = None

    def __init__(self, model, accuracy, steps=1001, batch_size=128,
                 loss = tf.nn.sigmoid_cross_entropy_with_logits,
                 optimizer = tf.train.AdamOptimizer):
        self.func_model = model
        self.batch_size = batch_size
        self.func_accuracy = accuracy
        self.steps = steps
        self.func_loss = loss
        self.func_optimizer = optimizer
        pass

    def fit(self, x_data, y_predict, test_data, test_labs):
        label_len = functools.reduce(np.dot, y_predict.shape[1:])
        image_hight = x_data.shape[1]
        image_width = x_data.shape[2]
        num_channels = x_data.shape[3]

        graph = tf.Graph()
        with graph.as_default():
            # Input data.
            drop = tf.placeholder_with_default(tf.constant(0.), None)
            tf_train_data = tf.placeholder(tf.float32)
            tf_train_labs = tf.placeholder(tf.float32, shape=(self.batch_size, label_len))
            tf_test_data = tf.constant(test_data)
            x_shaped = tf.reshape(tf_train_data, shape=[-1, image_hight, image_width, num_channels])

            # Training computation.
            with tf.variable_scope("predict") as scope:
                logits = self.func_model(x_shaped, drop=drop)
                scope.reuse_variables()
                test_prediction = self.func_model(tf_test_data, drop=None)

            loss = tf.reduce_mean(
                self.func_loss(logits, tf_train_labs))

            # Optimizer.
            optimizer = self.func_optimizer(0.001).minimize(loss)

        #########################

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            saver = tf.train.Saver()
            for step in range(self.steps):
                offset = (step * self.batch_size) % (y_predict.shape[0] - self.batch_size)
                batch_data = x_data[offset:(offset + self.batch_size), :, :, :]
                batch_labels = y_predict[offset:(offset + self.batch_size), :]
                feed_dict = {
                    tf_train_data: batch_data,
                    tf_train_labs: batch_labels.reshape(self.batch_size, label_len),
                    drop: 0.85,
                }
                _, l, train_p, test_p = session.run(
                    [optimizer, loss, logits, test_prediction], feed_dict=feed_dict)

                if (step % 50 == 0):
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % self.func_accuracy(train_p, batch_labels))
                    print('Test accuracy: %.1f%%' % self.func_accuracy(test_p, test_labs))
                    saver.save(session, "./first.model", global_step=step)
            print("Test accuracy: %.1f%%" % self.func_accuracy(test_p, test_labs))
        pass

    def predict(self, imgs):
        with tf.Session(graph=graph) as session:
            saver.restore(session, tf.train.latest_checkpoint('.'))
            predict = tf.argmax(tf.reshape(logits, [-1, num_seqlen, num_labels]), 2)
            return session.run(predict, feed_dict={tf_train_data: imgs})

