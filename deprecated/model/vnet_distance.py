# coding: utf8
import tensorflow as tf
from .models import *
from keras import backend as K
from ..archi.vnets import *


def dice_coef_multiD_tf(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred, axis=[0, 1]) + smooth)
    # sum over channels
    return tf.reduce_mean(dice)


def dice_loss_multi_D(y_true, y_pred):
    smooth = 1e-5
    return 1. - dice_coef_multiD_tf(y_true, y_pred, smooth)


class VnetDist(Model):

    def __init__(self, vnet, height=64, width=64, colors=3, n_classes=12, learning_rate=0.001, optimizer='Adam', model_path=None):

        # input dimension parameters
        self.h_input = height
        self.w_input = width
        self.channels_input = colors

        # placeholder definition
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.channels_input), name='input')
        self.Y = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, n_classes), name='ground_truth')

        self.lr = tf.get_variable("learning_rate", initializer=learning_rate, trainable=False)

        self.net = vnet

        self.Y_pred = vnet(self.X)

        self.loss = dice_loss_multi_D(self.Y, self.Y_pred)

        # optimization
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == 'RMS':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        # training procedures
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     self.training = self.optimizer.minimize(self.loss)
        self.training = self.optimizer.minimize(self.loss, var_list=self.net.trainable_weights)

        # At the end do what all models do with computation graph
        # computation graph
        self.saver = tf.train.Saver(var_list=self.net.trainable_weights)
        self.sess = tf.Session()
        # graph initialization
        # case where we update a previous graph
        if model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            print("\nLoading weights from a previous trained model at " + model_path + " !!!")
            self.saver.restore(self.sess, model_path)

        # hunt not-initialized variables
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def fit(self, x, y):

        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 1}
        _, lossval = self.sess.run([self.training, self.loss], feed_dict=feed_dict)
        return lossval

    def validate(self, x, y):

        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 0}
        lossval = self.sess.run([self.loss], feed_dict=feed_dict)
        return lossval

    def predict(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        y_pred = self.sess.run([self.Y_pred], feed_dict=feed_dict)
        return y_pred

    def close(self, path=None):

        if path is not None:
            self.saver.save(self.sess, path)
            print("\nmodel is saved at ", path, " !!!")
        self.sess.close()
        tf.reset_default_graph()

    def save(self, path):

        self.saver.save(self.sess, path)
        print("\nmodel is saved at ", path, " !!!")

    def __str__(self):

        description = '\narchitecture details\n'
        description += ('*' * len('architecture details') + '\n')
        description += (self.net.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')
        for w in self.net.trainable_weights:
            description += (str(w) + '\n')
        description += ('-' * len('trainable weights') + '\n')

        return description
