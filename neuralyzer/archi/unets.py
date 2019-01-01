# coding: utf8
import tensorflow as tf
from .bricks import *
from keras.layers import Conv2D


class Unet(Brick):

    def __init__(self,
                 brickname='unet',
                 layers=[2, 2, 2, 2, 2, 2, 2, 2],
                 filters=[64, 128, 256, 512, 512, 256, 128, 64],
                 kernels=[3, 3, 3, 3, 3, 3, 3, 3],
                 strides=[1, 1, 1, 1, 1, 1, 1, 1],
                 dropouts=[0., 0.25, 0.5, 0.5, 0.5, 0.5, 0.25, 0.],
                 activations=['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu'],
                 output_channels=1,
                 end_activation='tanh'):

        Brick.__init__(self, brickname)

        for depth in range(len(filters)):

            if depth < int(len(filters) / 2):

                encodername = "encoder_" + str(depth)
                encoderlayers = layers[depth]
                encoderfilters = filters[depth]
                encoderker = kernels[depth]
                encoderstride = strides[depth]
                encoderdropout = dropouts[depth]
                encoderac = activations[depth]

                self.ops.append(Encoding_Brick(brickname=encodername,
                                               filters=[encoderfilters for k in range(encoderlayers)],
                                               kernels=[encoderker for k in range(encoderlayers)],
                                               strides=[encoderstride for k in range(encoderlayers)],
                                               dropouts=[encoderdropout for k in range(encoderlayers)],
                                               activations=[encoderac for k in range(encoderlayers)]))
            else:

                decodername = "decoder_" + str(depth)
                decoderlayers = layers[depth]
                decoderfilters = filters[depth]
                decoderker = kernels[depth]
                decoderstride = strides[depth]
                decoderdropout = dropouts[depth]
                decoderac = activations[depth]

                self.ops.append(Decoding_Brick(brickname=decodername,
                                               filters=[decoderfilters for k in range(decoderlayers)],
                                               kernels=[decoderker for k in range(decoderlayers)],
                                               strides=[decoderstride for k in range(decoderlayers)],
                                               dropouts=[decoderdropout for k in range(decoderlayers)],
                                               activations=[decoderac for k in range(decoderlayers)]))

        if end_activation == 'tanh':
            self.ops.append(Conv2D(filters=output_channels,
                                   kernel_size=1,
                                   strides=1,
                                   padding='same',
                                   activation='tanh',
                                   name='final_convolution'))
        elif end_activation == 'sigmoid':
            self.ops.append(Conv2D(filters=output_channels,
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

        self.trainable_weights = self.weights()

    def __call__(self, arg_tensor):

        with tf.variable_scope(self.name, reuse=True):
            recalls = []
            y, yr = self.ops[0](arg_tensor)
            recalls.append(yr)

            for op in self.ops[1::]:

                if 'encoder' in op.name:

                    y, yr = op(y)
                    recalls.append(yr)

                elif 'decoder' in op.name:

                    # i.e. 'decoder' is in op.name
                    y = op(y, recalls[-1])
                    recalls.pop()
                else:
                    y = op(y)

            if not self.trainable_weights:
                self.trainable_weights = self.weights()

            return y
