import numpy as np
import tensorflow as tf
import functools


# create tf weight and biases var pair
def var(kernel_shape):
    """weights biases var init"""
    weight = tf.get_variable("weights", kernel_shape,
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable("biases", [kernel_shape[-1]],
                             initializer=tf.constant_initializer(0.0))
    return weight, biases


# create Conv Layer Model
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


# relu layer
def relu(x_input, kernel_shape, drop=None):
    """build relu layer"""
    weights, biases = var(kernel_shape)
    rtn = tf.nn.relu(tf.matmul(x_input, weights) + biases)
    if drop is not None and drop != 0.:
        rtn = tf.nn.dropout(rtn, drop)
    print(rtn)
    return rtn


# ts pack
class Learner:
    batch_size = 128
    func_model = None
    func_accuracy = None
    steps = 1001
    func_loss = None
    func_optimizer = None
    save_filename = 'default_learner'
    logits = None
    graph = None
    tf_train_data = None
    tf_drop = None

    def __init__(self, model, accuracy,
                 steps=1001, batch_size=128,
                 loss=tf.nn.sigmoid_cross_entropy_with_logits,
                 optimizer=tf.train.AdamOptimizer,
                 save='default_learner'):
        self.func_model = model
        self.batch_size = batch_size
        self.func_accuracy = accuracy
        self.steps = steps
        self.func_loss = loss
        self.func_optimizer = optimizer
        self.save_filename = save

    def fit(self, x_data, y_predict, vail_data, vail_labs):
        label_len = functools.reduce(np.dot, y_predict.shape[1:])
        image_height = x_data.shape[1]
        image_width = x_data.shape[2]
        num_channels = x_data.shape[3]

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data.
            self.tf_drop = tf.placeholder_with_default(tf.constant(0.), None, name='drop')
            self.tf_train_data = tf.placeholder(tf.float32, name='data')
            tf_train_labs = tf.placeholder(tf.float32, shape=(self.batch_size, label_len))
            tf_train_shaped = tf.reshape(self.tf_train_data,
                                         shape=[-1, image_height, image_width, num_channels])
            print(self.tf_train_data)

            tf_vail_data = tf.constant(vail_data)
            # tf.cond(tf.greater(tf.shape(tf_test_data)[0], 1),

            # Training computation.
            with tf.variable_scope("predict") as scope:
                self.logits = self.func_model(tf_train_shaped, drop=self.tf_drop)
                scope.reuse_variables()
                vail_prediction = self.func_model(tf_vail_data, drop=None)
                print(self.logits)

            loss = tf.reduce_mean(self.func_loss(self.logits, tf_train_labs))
            # Optimizer.
            optimizer = self.func_optimizer(0.001).minimize(loss)

        #########################

        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            saver = tf.train.Saver()
            for step in range(self.steps):
                offset = (step * self.batch_size) % (y_predict.shape[0] - self.batch_size)
                batch_data = x_data[offset:(offset + self.batch_size), :, :, :]
                batch_labels = y_predict[offset:(offset + self.batch_size), :]
                feed_dict = {
                    self.tf_train_data: batch_data,
                    tf_train_labs: batch_labels.reshape(self.batch_size, label_len),
                    self.tf_drop: 0.85,
                }
                _, l, train_p, vail_p = session.run(
                    [optimizer, loss, self.logits, vail_prediction], feed_dict=feed_dict)

                if step % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.1f%%' % self.func_accuracy(train_p, batch_labels))
                    print('Test accuracy: %.1f%%' % self.func_accuracy(vail_p, vail_labs))
                    saver.save(session, "./" + self.save_filename, global_step=step)
            print("Test accuracy: %.1f%%" % self.func_accuracy(vail_p, vail_labs))

    def save(self):
        pass

    def load(self, step=""):
        filename = self.save_filename
        if len(step) > 0:
            filename += '-' + step

        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default() as g:
            saver = tf.train.import_meta_graph(filename + ".meta")
            self.logits = g.get_tensor_by_name('predict/out/add:0')
            self.tf_train_data = g.get_tensor_by_name('data:0')
            self.tf_drop = g.get_tensor_by_name('drop:0')

        with tf.Session(graph=self.graph) as session:
            saver.restore(session, tf.train.latest_checkpoint('.'))

    def predict(self, imgs):
        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            return session.run(self.logits,
                               feed_dict={
                                   self.tf_train_data: imgs,
                                   tf_drop: 0.
                               })


