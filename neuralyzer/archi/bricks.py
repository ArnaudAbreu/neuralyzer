# coding: utf8
import tensorflow as tf
import os
import numpy as np
from keras.models import *
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, ConvLSTM2D, RepeatVector, Lambda, TimeDistributed, Deconvolution2D
from keras.layers import BatchNormalization, PReLU, SpatialDropout2D, Add, Activation, Conv2DTranspose
from keras import backend as K


class Brick:

    """
    Neuralizer elementary brick.
    """

    def __init__(self, brickname):

        """
        Arguments:
            - brickname: a string to create the namescope of the brick.
        """

        self.name = brickname
        self.ops = []
        self.trainable_weights = []

    def __str__(self):

        """
        Returns string representation of operations in the brick.
        """

        description = self.name + "\n"
        description += ("-" * len(self.name) + "\n")
        for op in self.ops:
            description += (op.__str__() + "\n")
        return description

    def weights(self):

        """
        Returns all the trainable weights in the brick.
        """

        w = []

        for op in self.ops:
            w += op.trainable_weights

        return w

    def __call__(self, arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=True):
            y = self.ops[0](arg_tensor)
            for op in self.ops[1::]:
                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y


class Encoding_Brick(Brick):

    """
    Unet elementary brick, for down operations.
    """

    def __init__(self, brickname, filters, kernels, strides, dropouts, activations):

        Brick.__init__(self, brickname)

        for k in range(len(kernels)):

            opname = "convolution_" + str(k)
            dropname = "dropout_" + str(k)
            opfilters = filters[k]
            opker = kernels[k]
            opstride = strides[k]
            opdropout = dropouts[k]
            opac = activations[k]

            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Now, I have all my convolutions, time for max pooling
        self.ops.append(MaxPooling2D(pool_size=(2, 2), name="pool"))

    def __call__(self, arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=True):
            y = Brick.__call__(self, arg_tensor)
            y_r = self.ops[-2](arg_tensor)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y, y_r


class Decoding_Brick(Brick):

    """
    Unet elementary brick, for up operations.
    """

    def __init__(self, brickname, filters, kernels, strides, dropouts, activations):

        Brick.__init__(self, brickname)
        # First I have to upsample input feature map
        self.ops.append(UpSampling2D(size=(2, 2), name="upsample"))

        for k in range(len(kernels)):

            # if k == 1, before convolving anything, I have to merge 2 tensors
            if k == 1:
                self.ops.append(Concatenate(axis=-1, name="concatenate"))

            opname = "convolution_" + str(k)
            dropname = "dropout_" + str(k)
            opfilters = filters[k]
            opker = kernels[k]
            opstride = strides[k]
            opdropout = dropouts[k]
            opac = activations[k]

            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

    def __call__(self, arg_tensor, recall_arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=True):
            # upsampling
            y = self.ops[0](arg_tensor)
            # first convolution
            y = self.ops[1](y)
            # first dropout
            y = self.ops[2](y)
            # merge
            y = self.ops[3]([recall_arg_tensor, y])
            for op in self.ops[4::]:
                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y


class Vnet_Starting_Brick(Brick):

    """
    Vnet elementary brick, for down0 operations.
    """

    def __init__(self, brickname, filters, kernel, dropout):

        Brick.__init__(self, brickname)

        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', use_bias=False))
        self.ops.append(BatchNormalization())
        self.ops.append(PReLU())
        self.jump = SpatialDropout2D(dropout)
        self.ops.append(self.jump)

    def __call__(self, arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            y = self.ops[0](arg_tensor)

            for op in self.ops[1::]:
                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y


class Vnet_Encoding_Brick(Brick):

    """
    Vnet elementary brick, for downk operations.
    """

    def __init__(self, brickname, filters, kernel, dropout, batchnorm=False):

        Brick.__init__(self, brickname)

        self.down = Conv2D(filters, 2, padding='same', kernel_initializer='he_normal', strides=(2, 2), activation='relu', name='conv1')
        self.ops.append(self.down)
        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', name='conv2'))
        if batchnorm:
            self.ops.append(BatchNormalization(name='bn1'))
        self.ops.append(Activation('relu'))
        self.ops.append(SpatialDropout2D(dropout, name='spatialdrop1'))
        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', name='conv3'))
        if batchnorm:
            self.ops.append(BatchNormalization(name='bn2'))
        self.ops.append(Activation('relu'))
        self.jump = SpatialDropout2D(dropout, name='spatialdrop2')
        self.ops.append(self.jump)

    def __call__(self, arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            if type(arg_tensor) is tuple:

                down, jump = arg_tensor
                m_down = Add()([down, jump])
                m_down = self.ops[0](m_down)

            else:

                m_down = self.ops[0](arg_tensor)

            m_jump = None

            for op in self.ops[1::]:

                if m_jump is None:
                    m_jump = op(m_down)
                else:
                    m_jump = op(m_jump)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return m_down, m_jump


class Vnet_Decoding_Brick(Brick):

    """
    Vnet elementary brick, for upk operations.
    """

    def __init__(self, brickname, filters, kernel, dropout):

        Brick.__init__(self, brickname)

        self.ops.append(Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='relu', name='deconv'))
        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', name='conv1'))
        self.ops.append(SpatialDropout2D(dropout, name='spatialdrop1'))
        self.ops.append(Activation('relu'))
        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', name='conv2'))
        self.ops.append(SpatialDropout2D(dropout, name='spatialdrop2'))
        self.ops.append(Activation('relu'))

    def __call__(self, arg_tensor, recall_arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            # arg_tensor is always a tuple, cause if it comes from encoder: down and jump
            # if it comes from decoder: up and jump

            x_downorup, x_jump = arg_tensor

            x = Add()([x_downorup, x_jump])

            up = self.ops[0](x)

            # recall_arg_tensor is always an encoding brick => tuple of tensors

            _, r_jump = recall_arg_tensor
            y = Add()([r_jump, up])

            for op in self.ops[1::]:

                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return up, y


class Vnet_Ending_Brick(Brick):

    """
    Vnet elementary brick, for upn operations.
    """

    def __init__(self, brickname, filters, kernel, dropout):

        Brick.__init__(self, brickname)

        self.ops.append(Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', name='deconv'))
        self.ops.append(PReLU())
        self.ops.append(SpatialDropout2D(dropout, name='spatialdrop1'))
        self.ops.append(Conv2D(filters, kernel, padding='same', kernel_initializer='he_normal', name='conv1'))
        self.ops.append(BatchNormalization(name='bn1'))
        self.ops.append(PReLU())
        self.ops.append(SpatialDropout2D(dropout, name='spatialdrop2'))

    def __call__(self, arg_tensor, recall_arg_tensor):

        """
        Apply brick operations to a given tensor.
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            # arg_tensor is always a tuple, cause it comes from decoder: up and jump

            x_up, x_jump = arg_tensor

            x = Add()([x_up, x_jump])

            up = self.ops[0](x)
            up = self.ops[1](up)

            # recall_arg_tensor is always an encoding brick => tuple of tensors
            # no! not if the encoding brick was the starting encoder

            if type(recall_arg_tensor) is tuple:
                _, r_jump = recall_arg_tensor
                y = Add()([r_jump, up])
            else:
                y = Add()([recall_arg_tensor, up])

            for op in self.ops[2::]:

                y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y
