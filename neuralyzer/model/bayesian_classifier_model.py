# coding: utf8
import tensorflow as tf
import tensorflow_probability as tfp
from .models import *
from keras import backend as K
from ..archi.bayesian_classifiers import *
import numpy

tfd = tfp.distributions


class BCLF(Model):

    def __init__(self, clf, height=28, width=28, colors=1, n_classes=10, learning_rate=0.001, kl_annealing=1., post_sampling=100, optimizer='SGD', model_path=None):

        """
        A bayesian classifier model made to fit on mnist-like datasets.
        """

        # input dimension parameters
        # design for the mnist dataset
        self.h_input = height
        self.w_input = width
        self.input_channels = colors
        self.n_classes = n_classes

        # placeholder definition
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.input_channels), name='input')
        # self.X_sample = [tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.input_channels), name='input_' + str(k)) for k in range(post_sampling)]
        self.Y = tf.placeholder(dtype=tf.int32, shape=(None,), name='ground_truth')
        self.lr = tf.get_variable("learning_rate", initializer=learning_rate, trainable=False)
        self.kl_annealing = tf.get_variable("kl_annealing", initializer=kl_annealing, trainable=False)
        self.post_sampling = post_sampling

        # architectures, clf here is an instance of BayesianClassifier
        self.bayesianlenet = clf

        # tensors for objectives

        # tensors for objectives
        # self.Y_pred = self.bayesianlenet(self.X)

        # Y_pred is a prediction formed on a single forward pass of the probabilistic layer
        # It is a cheap but noisy prediction.
        # Problem is that you cannot have a reliable prediction and a reliable
        # loss evaluation based on that prediction...
        # So, we draw multiple samples and average prediction.
        self.y_sample = []
        for i in range(post_sampling):
            self.y_sample.append(self.bayesianlenet(self.X))

        self.Y_pred = tf.reduce_mean(tf.stack(self.y_sample, axis=0), axis=0, keepdims=False)

        # self.logits = neural_net(images)
        self.labels_distribution = tfd.Categorical(logits=self.Y_pred)

        # Compute the -ELBO as the loss, averaged over the batch size.
        # neg_log_likelihood = -tf.reduce_mean(self.labels_distribution.log_prob(self.Y))
        # elbo_loss = neg_log_likelihood + kl

        self.Y_predcat = tf.cast(tf.argmax(self.Y_pred, axis=1, name='classes'), tf.int32)
        # self.Y_cat = tf.cast(tf.argmax(self.Y, axis=1, name='classes'), tf.int32)

        # objectives
        self.fY = tf.cast(self.Y, tf.float32)
        # self.loss = tf.reduce_mean(tf.keras.backend.categorical_crossentropy(self.fY, self.Y_pred, from_logits=False))
        self.neg_log_likelihood = -tf.reduce_mean(self.labels_distribution.log_prob(self.Y))
        self.kl = sum(self.bayesianlenet.losses) / tf.cast(tf.shape(self.Y)[0], tf.float32)
        self.loss = self.neg_log_likelihood + self.kl_annealing * self.kl
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.Y_predcat, self.Y), tf.float32))

        # optimization
        if optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif optimizer == 'RMS':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)

        # training procedures
        self.training = self.optimizer.minimize(self.loss, var_list=self.bayesianlenet.trainable_weights)

        # At the end do what all models do with computation graph
        # computation graph
        self.saver = tf.train.Saver(var_list=self.bayesianlenet.trainable_weights)
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

    def fit(self, x, y, kl):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.X: x, self.Y: y, self.kl_annealing: kl, K.learning_phase(): 1}
        # _, negloglikelyhood, kldiv, accuracyval = self.sess.run([self.training, self.neg_log_likelyhood, self.kl, self.accuracy], feed_dict=feed_dict)
        _, lossval, accuracyval = self.sess.run([self.training, self.loss, self.accuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def validate(self, x, y, kl):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)

        # In BCLF case, y is 'sparse', i.e. label is a unique integer for each class
        # accuracy will be computed here, not in tf graph...
        feed_dict = {self.X: x, self.Y: y, self.kl_annealing: kl, K.learning_phase(): 0}

        # probs = numpy.asarray([self.sess.run([self.labels_distribution.probs], feed_dict=feed_dict) for _ in range(self.post_sampling)])
        # mean_probs = numpy.mean(probs, axis=0)
        # print(mean_probs.shape)
        # mean probs is (batch, n_classes)
        # logprob = numpy.mean(numpy.log(mean_probs[numpy.arange(mean_probs.shape[0]), y.astype(int)]))
        # accuracyval = (numpy.argmax(mean_probs[0], axis=1) == y.astype(int)).mean()

        lossval, accuracyval = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return lossval, accuracyval

    def predict(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        # probs = numpy.asarray([self.sess.run([self.labels_distribution.probs], feed_dict=feed_dict) for _ in range(self.post_sampling)])
        # mean_probs = numpy.mean(probs, axis=0)
        predval = self.sess.run([self.Y_pred], feed_dict=feed_dict)
        return predval

    def sample_predict(self, x):

        feed_dict = {self.X: x, K.learning_phase(): 0}
        # probs = numpy.asarray([self.sess.run([self.labels_distribution.probs], feed_dict=feed_dict) for _ in range(self.post_sampling)])
        # mean_probs = numpy.mean(probs, axis=0)
        predvals = self.sess.run(self.y_sample, feed_dict=feed_dict)
        return predvals

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
        description += (self.bayesianlenet.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')
        for w in self.bayesianlenet.trainable_weights:
            description += (str(w) + '\n')
        description += ('-' * len('trainable weights') + '\n')

        return description
