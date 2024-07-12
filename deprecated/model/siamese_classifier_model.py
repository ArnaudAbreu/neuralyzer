# coding: utf8
import tensorflow as tf
from .models import *
from keras import backend as K
from ..archi.classifiers import *


class SiameseCLF(Model):

    def __init__(self, ref, dep, coeff=0.5, height=28, width=28, colors=1, n_classes=10, learning_rate=0.001, ref_path=None, dep_path=None, optimizer='Adam'):

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

        # architectures
        self.reference = ref
        self.dependency = dep

        # predictions
        self.Y_refpred = self.reference(self.X)
        self.Y_deppred = self.dependency(self.X)
        stacked_pred = tf.stack([self.Y_refpred, self.Y_deppred], axis=0)
        bary_pred = tf.reduce_mean(stacked_pred, 0)
        self.Y_ensemblepred = tf.nn.softmax(bary_pred)

        # categories
        # cast cause default might be tf.int64
        self.Y_cat = tf.cast(tf.argmax(self.Y, axis=1, name='classes'), tf.int32)
        self.Y_deppredcat = tf.cast(tf.argmax(self.Y_deppred, axis=1, name='depclasses'), tf.int32)
        self.Y_refpredcat = tf.cast(tf.argmax(self.Y_refpred, axis=1, name='refclasses'), tf.int32)
        self.Y_ensemblepredcat = tf.cast(tf.argmax(self.Y_ensemblepred, axis=1, name='ensembleclasses'), tf.int32)

        # objectives
        # defined to train dependency network
        # need ground_truth for classification
        self.floatY = tf.cast(self.Y, tf.float32)
        # need reference as one_hot to compute disagreement
        self.one_hot_Y_refpred = tf.one_hot(self.Y_refpredcat, n_classes)
        # logits = false, because output of network has been through softmax
        self.classifloss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.floatY, self.Y_deppred, from_logits=False))
        self.disagreementloss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.one_hot_Y_refpred, self.Y_deppred, from_logits=False))
        self.loss = self.classifloss - coeff * self.disagreementloss

        # accuracies
        self.refaccuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_refpredcat, self.Y_cat), tf.float32))
        self.depaccuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_deppredcat, self.Y_cat), tf.float32))
        self.ensembleaccuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_ensemblepredcat, self.Y_cat), tf.float32))

        # optimization
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == 'RMS':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        # training procedures
        self.training = self.optimizer.minimize(self.loss, var_list=self.dependency.trainable_weights)

        # At the end do what all models do with computation graph
        # computation graph
        self.refsaver = tf.train.Saver(var_list=self.reference.trainable_weights)
        self.depsaver = tf.train.Saver(var_list=self.dependency.trainable_weights)
        self.sess = tf.Session()

        # graph initialization
        if ref_path is not None:
            print("\nLoading weights from a previous REFERENCE trained model at " + ref_path + " !!!")
            self.refsaver.restore(self.sess, ref_path)
        if dep_path is not None:
            print("\nLoading weights from a previous DEPENDENCY trained model at " + dep_path + " !!!")
            self.depsaver.restore(self.sess, dep_path)

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
        _, lossval, accuracyval = self.sess.run([self.training, self.loss, self.depaccuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def validate(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 0}
        lossval, accuracyval = self.sess.run([self.loss, self.depaccuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def score_ensemble(self, x, y):

        feed_dict = {self.X: x, self.Y: y, K.learning_phase(): 0}
        refaccval, depaccval, accuracyval = self.sess.run([self.refaccuracy, self.depaccuracy, self.ensembleaccuracy], feed_dict=feed_dict)
        return refaccval, depaccval, accuracyval

    def predict(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        y_pred = self.sess.run([self.Y_ensemblepred], feed_dict=feed_dict)
        return y_pred

    def close(self, path=None):

        if path is not None:
            self.depsaver.save(self.sess, path)
            print("\nmodel is saved at ", path, " !!!")
        self.sess.close()
        tf.reset_default_graph()

    def save(self, path):
        self.depsaver.save(self.sess, path)
        print("\nmodel is saved at ", path, " !!!")

    def __str__(self):

        description = '\narchitecture details\n'
        description += ('*' * len('architecture details') + '\n')

        description += (self.reference.__str__() + '\n')
        description += (self.dependency.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')

        description += 'reference network:\n'
        description += ('-' * len('reference network:') + '\n')
        for w in self.reference.trainable_weights:
            description += (str(w) + '\n')

        description += '\ndependency network:\n'
        description += ('-' * len('dependency network:') + '\n')
        for w in self.dependency.trainable_weights:
            description += (str(w) + '\n')

        description += ('-' * len('trainable weights') + '\n')

        return description
