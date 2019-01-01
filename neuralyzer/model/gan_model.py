# coding: utf8
import tensorflow as tf
from .models import *
from keras import backend as K
from ..archi.unets import *
from ..archi.classifiers import *


class GAN(Model):

    def __init__(self, learning_rate=0.0002, lambdareg=10., model_path=None):

        """
        Let's call:
            - input image space X
            - output image space Y
            - generator Gy that generates Y space images
            - generator Gx that generates X space images
            - discriminator Dy that score Y space images
            - discriminator Dx that score X space images
        """

        # input dimension parameters
        self.h_input = 256
        self.w_input = 256
        self.channels_input = 3

        # placeholder definition
        self.real_X = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.channels_input), name='input')
        self.real_Y = tf.placeholder(dtype=tf.float32, shape=(None, self.h_input, self.w_input, self.channels_input), name='ground_truth')
        self.lr = tf.get_variable("learning_rate", initializer=learning_rate, trainable=False)
        self.lambdareg = tf.get_variable("lambda_reg", initializer=lambdareg, trainable=False)

        # architectures
        self.generatorx = Unet(brickname='generatorx', output_channels=3)
        self.generatory = Unet(brickname='generatory', output_channels=3)
        self.discriminatorx = Fully_Convolutional_Classifier(brickname='discriminatorx')
        self.discriminatory = Fully_Convolutional_Classifier(brickname='discriminatory')

        # tensors for objectives
        self.fakey = self.generatory(self.real_X)
        self.fakex = self.generatorx(self.real_Y)
        self.cycle_fakey = self.generatory(self.fakex)
        self.cycle_fakex = self.generatorx(self.fakey)

        self.is_fakex = self.discriminatorx(self.fakex)
        self.is_fakey = self.discriminatory(self.fakey)

        self.is_realx = self.discriminatorx(self.real_X)
        self.is_realy = self.discriminatory(self.real_Y)

        # objectives

        # Dy must predict 1 for real image
        self.loss_recognize_realy = tf.reduce_mean((self.is_realy - tf.ones_like(self.is_realy)) ** 2)
        # Dy must predict 0 for fake image
        self.loss_recognize_fakey = tf.reduce_mean((self.is_fakey - tf.zeros_like(self.is_fakey)) ** 2)
        # Dy total loss
        self.loss_disc_y = (self.loss_recognize_fakey + self.loss_recognize_realy) / 2

        # Dx must predict 1 for real image
        self.loss_recognize_realx = tf.reduce_mean((self.is_realx - tf.ones_like(self.is_realx)) ** 2)
        # Dx must predict 0 for fake image
        self.loss_recognize_fakex = tf.reduce_mean((self.is_fakex - tf.zeros_like(self.is_fakex)) ** 2)
        # Dx total loss
        self.loss_disc_x = (self.loss_recognize_fakex + self.loss_recognize_realx) / 2

        # Final D loss
        self.loss_disc_total = self.loss_disc_x + self.loss_disc_y

        # Gy must produce good Y images in the sence of Dy
        self.loss_produce_fakey = tf.reduce_mean((self.is_fakey - tf.ones_like(self.is_fakey)) ** 2)

        # Gx must produce good X images in the sence of Dx
        self.loss_produce_fakex = tf.reduce_mean((self.is_fakex - tf.ones_like(self.is_fakex)) ** 2)

        # cycle consistency...
        self.loss_cycle = tf.reduce_mean(tf.abs(self.real_Y - self.cycle_fakey)) + tf.reduce_mean(tf.abs(self.real_X - self.cycle_fakex))

        # Final G loss
        self.loss_generator_total = self.loss_produce_fakey + self.loss_produce_fakex + (self.lambdareg * self.loss_cycle)

        # optimization
        self.optimizer = tf.train.AdamOptimizer(self.lr)

        # training procedures
        self.G_training = self.optimizer.minimize(self.loss_generator_total, var_list=self.generatorx.trainable_weights + self.generatory.trainable_weights)
        self.D_training = self.optimizer.minimize(self.loss_disc_total, var_list=self.discriminatorx.trainable_weights + self.discriminatory.trainable_weights)

        # At the end do what all models do with computation graph
        # computation graph
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        # graph initialization
        # case where we update a previous graph
        if model_path is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            print("\nLoading weights from a previous trained model at " + model_path + " !!!")
            self.saver.restore(self.sess, model_path)

    def fit(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.real_X: x, self.real_Y: y, K.learning_phase(): 1}
        _, lossvalG = self.sess.run([self.G_training, self.loss_generator_total], feed_dict=feed_dict)
        _, lossvalD = self.sess.run([self.D_training, self.loss_disc_total], feed_dict=feed_dict)
        return lossvalG, lossvalD

    def validate(self, x, y):

        # In the original ne_ne implementation, no return statement.
        # I changed it to be clearer (my point of view...)
        feed_dict = {self.real_X: x, self.real_Y: y, K.learning_phase(): 0}
        lossvalG, lossvalD = self.sess.run([self.loss_generator_total, self.loss_disc_total], feed_dict=feed_dict)
        return lossvalG, lossvalD

    def test_y_generator(self, x):

        feed_dict = {self.real_X: x, K.learning_phase(): 0}
        fakey = self.sess.run([self.fakey], feed_dict=feed_dict)
        return fakey

    def test_x_generator(self, y):

        feed_dict = {self.real_Y: y, K.learning_phase(): 0}
        fakex = self.sess.run([self.fakex], feed_dict=feed_dict)
        return fakex

    def __str__(self):

        description = '\narchitecture details\n'
        description += ('*' * len('architecture details') + '\n')
        description += (self.generatorx.__str__() + '\n')
        description += (self.generatory.__str__() + '\n')
        description += (self.discriminatorx.__str__() + '\n')
        description += (self.discriminatory.__str__() + '\n')

        description += '\ntrainable weights\n'
        description += ('*' * len('trainable weights') + '\n')
        description += (str(self.generatorx.trainable_weights) + '\n')
        description += ('-' * len('trainable weights') + '\n')
        description += (str(self.generatory.trainable_weights) + '\n')
        description += ('-' * len('trainable weights') + '\n')
        description += (str(self.discriminatorx.trainable_weights) + '\n')
        description += ('-' * len('trainable weights') + '\n')
        description += (str(self.discriminatory.trainable_weights) + '\n')
        description += ('-' * len('trainable weights') + '\n')

        return description
