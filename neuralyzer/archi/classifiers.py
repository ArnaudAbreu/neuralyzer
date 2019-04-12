# coding: utf8
import tensorflow as tf
from .bricks import *
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


class Fully_Convolutional_Classifier(Brick):

    def __init__(self,
                 brickname='classifier',
                 filters=[64, 128, 256, 512],
                 kernels=[3, 3, 3, 3],
                 strides=[1, 1, 1, 1],
                 dropouts=[0., 0.1, 0.2, 0.3],
                 activations=['relu', 'relu', 'relu', 'relu'],
                 output_channels=1,
                 end_activation='sigmoid'):

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = activations[depth]
            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(Dropout(rate=opdropout, name=dropname))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name="pool"))

        if end_activation == 'tanh':
            self.ops.append(Conv2D(filters=1,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   activation='tanh',
                                   name='final_convolution'))
        elif end_activation == 'sigmoid':
            self.ops.append(Conv2D(filters=1,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   activation='sigmoid',
                                   name='final_convolution'))
        elif end_activation == 'softmax':
            assert output_channels > 1
            self.ops.append(Conv2D(filters=output_channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   activation='softmax',
                                   name='final_convolution'))


class Classifier(Brick):

    def __init__(self,
                 brickname='classifier',
                 filters=[20, 50],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dropouts=[0., 0.],
                 fc=[500],
                 fcdropouts=[0.],
                 conv_activations=['relu', 'relu'],
                 fc_activations=['relu'],
                 end_activation='softmax',
                 output_channels=10):

        """
        LeNet classical classifier.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = conv_activations[depth]
            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name="pool"))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(Flatten())

        for depth in range(len(fc)):

            opname = 'fc_' + str(depth)
            dropname = 'fc_dropout_' + str(depth)
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


class ClassifierAccumulator(Brick):
    def __init__(self,
                 brickname='classifier',
                 filters=[20, 50],
                 kernels=[5, 5],
                 strides=[1, 1],
                 dropouts=[0., 0.],
                 fc=[500],
                 fcdropouts=[0.],
                 conv_activations=['relu', 'relu'],
                 fc_activations=['relu'],
                 end_activation='softmax',
                 output_channels=10):

        """
        classifier accumulator for stochastic weight averaging.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "convolution_" + str(depth)
            dropname = "dropout_" + str(depth)
            poolname = "pool_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opdropout = dropouts[depth]
            opac = conv_activations[depth]
            self.ops.append(Conv2D(filters=opfilters,
                                   kernel_size=opker,
                                   strides=(opstride, opstride),
                                   activation=opac,
                                   padding='same',
                                   name=opname))
            self.ops.append(MaxPooling2D(pool_size=(2, 2), name="pool"))
            self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(Flatten())

        for depth in range(len(fc)):

            opname = 'fc_' + str(depth)
            dropname = 'fc_dropout_' + str(depth)
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


class MultiResClassifier(Brick):

    def __init__(self, levels, depth, filters, kernels, activations, brickname='classifier'):

        Brick.__init__(self, brickname=brickname)

        self.encoders = dict()

        for level in levels:
            self.encoders[level] = Encoder('encoder_' + str(level), depth, filters, kernels, activations)

        self.ops = [self.encoders[level] for level in levels]

    def __call__(self, x):

        """
        In that particular case, x is a dictionary and its keys are levels.
        """

        # x[level] should be a list of placeholders
        ylist = []

        for level in self.encoders.keys():
            inputs = x[level]
            ylocallist = [self.encoders[level](xin) for xin in inputs]
            ylist += ylocallist

        y = tf.concat(ylist, -1)

        if not self.trainable_weights:
            self.trainable_weights = self.weights()

        return y
