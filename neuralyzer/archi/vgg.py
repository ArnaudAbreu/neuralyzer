# coding: utf8
import tensorflow as tf
from .bricks import *
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras import regularizers


class VGG(Brick):

    def __init__(self,
                 brickname='VGG16',
                 filters=[48, 96, 192],
                 kernels=[3, 3, 3],
                 strides=[1, 1, 1],
                 dropouts=[0.25, 0.25, 0.25],
                 thicknesses=[2, 2, 2],
                 fc=[512, 256],
                 fcdropouts=[0.5, 0.5],
                 conv_activations=['relu', 'relu', 'relu'],
                 fc_activations=['relu', 'relu'],
                 end_activation='softmax',
                 output_channels=10):

        """
        VGG classical classifier.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            bnname = "bn_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = conv_activations[depth]
            thickness = thicknesses[depth]
            weight_decay = 0.0005
            for t in range(thickness):
                self.ops.append(Conv2D(filters=opfilters,
                                       kernel_size=opker,
                                       strides=(opstride, opstride),
                                       activation=opac,
                                       padding='same',
                                       name=opname + '_' + str(t)))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name=poolname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(Flatten())

        for depth in range(len(fc)):

            opname = 'fc_' + str(depth)
            dropname = 'fc_dropout_' + str(depth)
            bnname = 'fc_bn_' + str(depth)
            opunits = fc[depth]
            opdropout = fcdropouts[depth]
            opac = fc_activations[depth]
            self.ops.append(Dense(opunits, activation=opac, name=opname))
            self.ops.append(Dropout(opdropout, name=dropname))

        self.ops.append(Dense(output_channels,
                              activation=end_activation,
                              name='final_fc'))

    def transfer(self, other_clf):

        op_list = []

        for w, wother in zip(self.trainable_weights, other_clf.trainable_weights):

            op_list.append(w.assign(wother))

        return op_list


class VGGAccumulator(Brick):

    def __init__(self,
                 brickname='VGG16',
                 filters=[48, 96, 192],
                 kernels=[3, 3, 3],
                 strides=[1, 1, 1],
                 dropouts=[0.25, 0.25, 0.25],
                 thicknesses=[2, 2, 2],
                 fc=[512, 256],
                 fcdropouts=[0.5, 0.5],
                 conv_activations=['relu', 'relu', 'relu'],
                 fc_activations=['relu', 'relu'],
                 end_activation='softmax',
                 output_channels=10):

        """
        VGG classical classifier.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            bnname = "bn_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = conv_activations[depth]
            thickness = thicknesses[depth]
            weight_decay = 0.0005
            for t in range(thickness):
                self.ops.append(Conv2D(filters=opfilters,
                                       kernel_size=opker,
                                       strides=(opstride, opstride),
                                       activation=opac,
                                       padding='same',
                                       name=opname + '_' + str(t)))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name=poolname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(Flatten())

        for depth in range(len(fc)):

            opname = 'fc_' + str(depth)
            dropname = 'fc_dropout_' + str(depth)
            bnname = 'fc_bn_' + str(depth)
            opunits = fc[depth]
            opdropout = fcdropouts[depth]
            opac = fc_activations[depth]
            self.ops.append(Dense(opunits, activation=opac, name=opname))
            self.ops.append(Dropout(opdropout, name=dropname))

        self.ops.append(Dense(output_channels,
                              activation=end_activation,
                              name='final_fc'))

    def accumulate(self, global_step, explorer):

        op_list = []

        for wswa, w in zip(self.trainable_weights, explorer.trainable_weights):

            op_list.append(wswa.assign((wswa * global_step + w) / (global_step + 1)))

        return op_list

    def initialize(self, explorer):

        op_list = []

        for wswa, w in zip(self.trainable_weights, explorer.trainable_weights):

            op_list.append(wswa.assign(w))

        return op_list
