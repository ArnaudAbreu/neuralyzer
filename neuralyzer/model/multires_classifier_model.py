# coding: utf8
import tensorflow as tf
from .models import *
from keras import backend as K
import keras
from ..archi.classifiers import *

# this time I'm using a keras model object (just like Vincent)


def head_classifier(Y, nbClasses):
    return Conv2D(nbClasses, 1, activation='softmax')(Y)


def make_model(levels, filters, kernels, activations, h, w, c, n_classes):

    depth = len(filters)

    clf = MultiResClassifier(levels, depth, filters, kernels, activations)

    X = dict()

    inputnumbers = [4**k for k in range(len(levels))]

    for k in range(len(levels)):
        inputnumber = inputnumbers[3 - k]
        level = levels[k]
        X[level] = [tf.placeholder(dtype=tf.float32, shape=(None, h, w, c)) for n in range(inputnumber)]

    Y = clf(X)

    output = head_classifier(Y, n_classes)

    loss = dice_loss_multi_D

    memo_train["loss"] = "dice_loss_multi_D"

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss=loss)
    return model, loss
