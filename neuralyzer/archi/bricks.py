# coding: utf8
import tensorflow as tf
import os
import numpy as np
from keras.models import *
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, ConvLSTM2D, RepeatVector, Lambda, TimeDistributed, Deconvolution2D
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
