# coding: utf8
from keras.layers import Input, Concatenate, concatenate, Conv2D, Dropout, BatchNormalization, ELU, Activation, PReLU, SpatialDropout2D, LeakyReLU, Multiply, Subtract, Conv2DTranspose, Add
import tensorflow as tf
from .bricks import *
import math


class Vnet(Brick):

    def __init__(self,
                 brickname='vnet',
                 filters=[16, 32, 64, 128, 64, 32, 16],
                 kernels=[5, 5, 5, 5, 5, 5, 5],
                 dropouts=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                 head='distance',
                 n_classes=12):

        Brick.__init__(self, brickname)
        self.headname = head
        self.n_classes = n_classes

        for k in range(len(filters)):

            n_filters = filters[k]
            kernel_size = kernels[k]
            drop = dropouts[k]

            if k < int(math.floor(len(filters) / 2.)):
                name = 'encoder_' + str(k)
                if k == 0:
                    # starting block case
                    self.ops.append(Vnet_Starting_Brick(name, filters=n_filters, kernel=kernel_size, dropout=drop))
                else:
                    # encoding block case
                    self.ops.append(Vnet_Encoding_Brick(name, filters=n_filters, kernel=kernel_size, dropout=drop))

            elif k == int(math.floor(len(filters) / 2.)):
                name = 'last_encoder_' + str(k)
                # last encoding block case, difference is only batchnorm
                self.ops.append(Vnet_Encoding_Brick(name, filters=n_filters, kernel=kernel_size, dropout=drop, batchnorm=True))

            elif k == len(filters) - 1:
                name = 'decoder_' + str(k)
                # end block case
                self.ops.append(Vnet_Ending_Brick(name, filters=n_filters, kernel=kernel_size, dropout=drop))

            else:
                name = 'decoder_' + str(k)
                # decoding block
                self.ops.append(Vnet_Decoding_Brick(name, filters=n_filters, kernel=kernel_size, dropout=drop))

    def __call__(self, arg_tensor):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # initialize recall tensor list
            recalls = []
            # apply first block, it's a starting block:
            # 1 input, 1 output
            y = self.ops[0](arg_tensor)
            if type(y) is tuple:
                print(y[0].name)
                print(y[0].shape)
                print(y[1].name)
                print(y[1].shape)
                print('#' * 20)
            else:
                print(y.name)
                print(y.shape)
                print('#' * 20)
            # output must be stored for later recall
            recalls.append(y)

            for op in self.ops[1:-1]:

                if 'encoder' in op.name:
                    # 2 outputs, can have tuple or single input
                    y = op(y)
                    if type(y) is tuple:
                        print(y[0].name)
                        print(y[0].shape)
                        print(y[1].name)
                        print(y[1].shape)
                        print('#' * 20)
                    else:
                        print(y.name)
                        print(y.shape)
                        print('#' * 20)
                    if 'last' not in op.name:
                        recalls.append(y)

                elif 'decoder' in op.name:

                    # i.e. 'decoder' is in op.name
                    # 2 inputs, 2 recall, 2 outputs (2 inputs hidden in single tuple)
                    y = op(y, recalls[-1])
                    if type(y) is tuple:
                        print(y[0].name)
                        print(y[0].shape)
                        print(y[1].name)
                        print(y[1].shape)
                        print('#' * 20)
                    else:
                        print(y.name)
                        print(y.shape)
                        print('#' * 20)
                    recalls.pop()

            y = self.ops[-1](y, recalls[-1])

            self.head = Conv2D(self.n_classes, 1, activation='sigmoid')
            y = self.head(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()
                self.trainable_weights.append(self.head)

            return y
