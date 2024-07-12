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
                self.trainable_weights += self.head.trainable_weights

            return y


class Vnet_4levels:

    def __init__(self, dropout_rate):

        self.deph = 16
        self.batch_norm = True
        self.padding = "same"
        self.dropout_rate = dropout_rate
        self.spatialdrop = True

    def __call__(self, inputs):

        # PATCH_DIM
        Y = Conv2D(self.deph, 5, padding=self.padding, kernel_initializer='he_normal', use_bias=False)(inputs)
        Y = BatchNormalization()(Y)
        Y = PReLU()(Y)
        jump1 = self.drop(Y)

        # PATCH_DIM/2
        Y = Conv2D(self.deph * 2, 2, padding=self.padding, kernel_initializer='he_normal', strides=(2, 2))(jump1)
        Down = Activation('relu')(Y)
        Y = Conv2D(self.deph * 2, 5, padding=self.padding, kernel_initializer='he_normal')(Down)
        Y = Activation('relu')(Y)
        Y = self.drop(Y)
        Y = Conv2D(self.deph * 2, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = Activation('relu')(Y)
        jump2 = self.drop(Y)

        # PATCH_DIM/4
        Y = Conv2D(self.deph * 4, 2, padding=self.padding, kernel_initializer='he_normal', strides=(2, 2))(Add()([Down, jump2]))
        Down = Activation('relu')(Y)
        Y = Conv2D(self.deph * 4, 5, padding=self.padding, kernel_initializer='he_normal')(Down)
        Y = Activation('relu')(Y)
        Y = self.drop(Y)
        Y = Conv2D(self.deph * 4, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = self.drop(Y)
        jump3 = Activation('relu')(Y)

        # PATCH_DIM/8
        Y = Conv2D(self.deph * 8, 2, padding=self.padding, kernel_initializer='he_normal', strides=(2, 2))(Add()([Down, jump3]))
        Down = Activation('relu')(Y)

        Y = Conv2D(self.deph * 8, 5, padding=self.padding, kernel_initializer='he_normal')(Down)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = self.drop(Y)
        Y = Conv2D(self.deph * 8, 5, padding=self.padding, kernel_initializer='he_normal', use_bias=False)(Y)
        Y = BatchNormalization()(Y)
        Y = Activation('relu')(Y)
        Y = self.drop(Y)
        Y = Add()([Down, Y])

        # PATCH_DIM/4
        Y = Conv2DTranspose(filters=self.deph * 4, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(Y)
        Up = Activation('relu')(Y)

        Y = Add()([jump3, Up])

        Y = Conv2D(self.deph * 4, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = self.drop(Y)
        Y = Activation('relu')(Y)
        Y = Conv2D(self.deph * 4, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = self.drop(Y)
        Y = Activation('relu')(Y)

        Y = Add()([Y, Up])

        # PATCH_DIM/2
        Y = Conv2DTranspose(filters=self.deph * 2, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(Y)
        Up = Activation('relu')(Y)

        Y = Add()([jump2, Up])

        Y = Conv2D(self.deph * 2, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = self.drop(Y)
        Y = Activation('relu')(Y)
        Y = Conv2D(self.deph * 2, 5, padding=self.padding, kernel_initializer='he_normal')(Y)
        Y = self.drop(Y)
        Y = Activation('relu')(Y)

        Y = Add()([Y, Up])

        # PATCH_DIM
        Y = Conv2DTranspose(filters=self.deph, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal')(Y)
        Y = PReLU()(Y)
        Y = Add()([jump1, Y])
        Y = self.drop(Y)
        Y = Conv2D(self.deph, 5, padding=self.padding, kernel_initializer='he_normal', use_bias=False)(Y)
        Y = BatchNormalization()(Y)
        Y = PReLU()(Y)
        Y = self.drop(Y)
        # Y = Conv2D(1, 1, activation = 'sigmoid')(Y)

        return Y

    def drop(self, Y):
        if self.spatialdrop:
            return SpatialDropout2D(self.dropout_rate)(Y)
        else:
            return Dropout(self.dropout_rate)(Y)
