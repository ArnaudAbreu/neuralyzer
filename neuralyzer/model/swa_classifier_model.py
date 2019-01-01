# coding: utf8
import tensorflow as tf
from .models import *
from keras import backend as K
from ..archi.classifiers import *


class SWACLF(Model):

    def __init__(self, clf, acc, height=28, width=28, colors=1, n_classes=10, learning_rate=0.001, optimizer='Adam', explorer_path=None):

        """
        A classifier model made to fit on mnist-like dataset.
        """

        # input dimension parameters
        # design for the mnist dataset
        self.h_input = height
        self.w_input = width
        self.input_channels = colors
        self.n_classes = n_classes

        # placeholder definition
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.input_channels), name='input')
        self.Y = tf.placeholder(dtype=tf.int32, shape=(None, n_classes), name='ground_truth')
        self.lr = tf.get_variable("learning_rate", initializer=learning_rate, trainable=False)
        self.global_step = tf.placeholder(dtype=tf.float32)

        # architectures
        self.lenet = clf
        self.accumulator = acc

        # tensors for objectives
        self.Y_pred = self.lenet(self.X)
        self.Y_predcat = tf.cast(tf.argmax(self.Y_pred, axis=1, name='classes'), tf.int32)
        self.Y_cat = tf.cast(tf.argmax(self.Y, axis=1, name='classes'), tf.int32)

        self.Y_accpred = self.accumulator(self.X)
        self.Y_accpredcat = tf.cast(tf.argmax(self.Y_accpred, axis=1, name='classes'), tf.int32)

        # objectives
        self.fY = tf.cast(self.Y, tf.float32)
        self.loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.fY, self.Y_pred, from_logits=False))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_predcat, self.Y_cat), tf.float32))
        self.accaccuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_accpredcat, self.Y_cat), tf.float32))

        # optimization
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == 'RMS':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        # training procedures
        self.training = self.optimizer.minimize(self.loss, var_list=self.lenet.trainable_weights)
        self.accumulation = self.accumulator.accumulate(self.global_step, self.lenet)
        self.accinit = self.accumulator.initialize(self.lenet)

        # At the end do what all models do with computation graph
        # computation graph
        self.expsaver = tf.train.Saver(var_list=self.lenet.trainable_weights)
        self.accsaver = tf.train.Saver(var_list=self.accumulator.trainable_weights)
        self.sess = tf.Session()
        # graph initialization
        # case where we update a previous graph
        if explorer_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            print("\nLoading weights from a previous trained model at " + explorer_path + " !!!")
            self.expsaver.restore(self.sess, explorer_path)

        # hunt not-initialized variables
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def fit(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 1}
        _, lossval, accuracyval = self.sess.run([self.training, self.loss, self.accuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def accumulate(self, step):

        self.sess.run([self.accumulation], feed_dict={self.global_step: step})

    def init_acc(self):

        self.sess.run([self.accinit])

    def validate(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 0}
        lossval, accuracyval = self.sess.run([self.accuracy, self.accaccuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def test(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        y_pred = self.sess.run([self.Y_accpred], feed_dict=feed_dict)
        return y_pred

    def close(self, path=None):

        if path is not None:
            self.accsaver.save(self.sess, path)
            print("\nmodel is saved at ", path, " !!!")
        self.sess.close()
        tf.reset_default_graph()

    def __str__(self):

        description = '\narchitecture details\n'
        description += ('*' * len('architecture details') + '\n')
        description += (self.lenet.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')
        for w in self.lenet.trainable_weights:
            description += (str(w) + '\n')
        description += ('-' * len('trainable weights') + '\n')

        return description


class ConstSWACLF(Model):

    def __init__(self, clf, acc, coeff=0.5, height=28, width=28, colors=1, n_classes=10, learning_rate=0.001, optimizer='Adam', explorer_path=None):

        """
        A classifier model made to fit on mnist-like dataset.
        """

        # input dimension parameters
        # design for the mnist dataset
        self.h_input = height
        self.w_input = width
        self.input_channels = colors
        self.n_classes = n_classes

        # placeholder definition
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.input_channels), name='input')
        self.Y = tf.placeholder(dtype=tf.int32, shape=(None, n_classes), name='ground_truth')
        self.lr = tf.get_variable("learning_rate", initializer=learning_rate, trainable=False)
        self.global_step = tf.placeholder(dtype=tf.float32)

        # architectures
        self.lenet = clf
        self.accumulator = acc

        # tensors for objectives
        self.Y_pred = self.lenet(self.X)
        self.Y_predcat = tf.cast(tf.argmax(self.Y_pred, axis=1, name='expclasses'), tf.int32)
        self.Y_cat = tf.cast(tf.argmax(self.Y, axis=1, name='classes'), tf.int32)

        self.Y_accpred = self.accumulator(self.X)
        self.Y_accpredcat = tf.cast(tf.argmax(self.Y_accpred, axis=1, name='accclasses'), tf.int32)

        self.one_hot_Y_accpred = tf.one_hot(self.Y_accpredcat, n_classes)

        # objectives
        self.fY = tf.cast(self.Y, tf.float32)

        self.classifloss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.fY, self.Y_pred, from_logits=False))
        self.disagreementloss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.one_hot_Y_accpred, self.Y_pred, from_logits=False))
        self.loss = self.classifloss - coeff * self.disagreementloss

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_predcat, self.Y_cat), tf.float32))
        self.accaccuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_accpredcat, self.Y_cat), tf.float32))

        # optimization
        # optimization
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == 'RMS':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        # training procedures
        self.training = self.optimizer.minimize(self.loss, var_list=self.lenet.trainable_weights)
        self.accumulation = self.accumulator.accumulate(self.global_step, self.lenet)
        self.accinit = self.accumulator.initialize(self.lenet)

        # At the end do what all models do with computation graph
        # computation graph
        self.expsaver = tf.train.Saver(var_list=self.lenet.trainable_weights)
        self.accsaver = tf.train.Saver(var_list=self.accumulator.trainable_weights)
        self.sess = tf.Session()
        # graph initialization
        # case where we update a previous graph
        if explorer_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            print("\nLoading weights from a previous trained model at " + explorer_path + " !!!")
            self.expsaver.restore(self.sess, explorer_path)

        # hunt not-initialized variables
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def fit(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 1}
        _, lossval, accuracyval = self.sess.run([self.training, self.classifloss, self.accuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def accumulate(self, step):

        self.sess.run([self.accumulation], feed_dict={self.global_step: step})

    def init_acc(self):

        self.sess.run([self.accinit])

    def validate(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 0}
        lossval, accuracyval = self.sess.run([self.accuracy, self.accaccuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def test(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        y_pred = self.sess.run([self.Y_accpred], feed_dict=feed_dict)
        return y_pred

    def close(self, path=None):

        if path is not None:
            self.accsaver.save(self.sess, path)
            print("\nmodel is saved at ", path, " !!!")
        self.sess.close()
        tf.reset_default_graph()

    def __str__(self):

        description = '\narchitecture details\n'
        description += ('*' * len('architecture details') + '\n')
        description += (self.lenet.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')
        for w in self.lenet.trainable_weights:
            description += (str(w) + '\n')
        description += ('-' * len('trainable weights') + '\n')

        return description
