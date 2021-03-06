# coding: utf8
import tensorflow as tf
import tensorflow_probability as tfp
from .bricks import *
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


class BayesianClassifier(Brick):

    def __init__(self,
                 brickname='classifier',
                 filters=[20, 50],
                 kernels=[5, 5],
                 strides=[1, 1],
                 fc=[500],
                 conv_activations=['relu', 'relu'],
                 fc_activations=['relu'],
                 end_activation=None,
                 output_channels=10):

        """
        LeNet classical classifier, Bayesian version.
        """

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            opname = "bayesianconvolution_" + str(depth)
            poolname = "pool_" + str(depth)
            opfilters = filters[depth]
            opker = kernels[depth]
            opstride = strides[depth]
            opac = conv_activations[depth]
            self.ops.append(tfp.layers.Convolution2DFlipout(filters=opfilters,
                                                            kernel_size=opker,
                                                            strides=(opstride, opstride),
                                                            activation=opac,
                                                            padding='same',
                                                            name=opname))
            self.ops.append(tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool"))
            # Dropout is not necessary with bayesian networks (not supposed to over-fit)
            # self.ops.append(Dropout(rate=opdropout, name=dropname))

        # Before applying fully connected layers, I have to flatten
        self.ops.append(tf.layers.Flatten())

        for depth in range(len(fc)):

            opname = 'bayesianfc_' + str(depth)
            opunits = fc[depth]
            opac = fc_activations[depth]
            self.ops.append(tfp.layers.DenseFlipout(opunits, activation=opac, name=opname))
            # Dropout is not necessary with bayesian networks (not supposed to over-fit)
            # self.ops.append(Dropout(opdropout, name=dropname))

        self.ops.append(tfp.layers.DenseFlipout(output_channels,
                                                activation=end_activation,
                                                name='bayesianfinal_fc'))
        self.losses = []
        self.called = False

    def __call__(self, arg_tensor):

        output = Brick.__call__(self, arg_tensor)

        if not self.called:
            for operation in self.ops:
                if 'bayesian' in operation.name:
                    if type(operation.losses) is list:
                        self.losses += operation.losses
                    else:
                        self.losses.append(operation.losses)
            self.called = True

        return output

    def transfer(self, other_clf):

        op_list = []

        for w, wother in zip(self.trainable_weights, other_clf.trainable_weights):

            op_list.append(w.assign(wother))

        return op_list
