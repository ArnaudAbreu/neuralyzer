# coding: utf8
from ..data import FolderGenerator as foldgen
from ..model import BCLF
from ..archi import BayesianClassifier
from ..render import monitor
from tqdm import tqdm
import numpy
import argparse
import os
import pickle


def train(train_folder, valid_folder, batchsize, patchsize, inputchannels, epochs, device, lr, opt, outfolder, post_sampling):

    """
    A function to train a CNN to classify lymph nodes in 2 categories:
    Follicular Hyperplasia or Follicular Lymphoma

    Arguments:
        - train_folder: str, path to train dataset.
        - valid_folder: str, path to test dataset.
        - batchsize: int, batch size used to fit on one iteration.
        - patchsize: int, size of images (side in pixels).
        - inputchannels: int, number of channels in images (usually 3).
        - epochs: int, number of fits on the entire training set.
        - device: str, '0' or '1' GPU card id.
        - lr: float, learning rate used for fitting.
        - opt: str, tf optimizer used 'Adam', 'SGD', ...
        - outfolder: str, path to folder where to store network and plots.
        - post_sampling: int, number of samples to draw to compute reliable monte carlo gradients.

    Returns:
        - Nothing, just create, train and store network.
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    h = patchsize
    w = patchsize
    c = inputchannels

    train_data = foldgen(train_folder, batchsize, size=(h, w), class_mode='sparse')
    valid_data = foldgen(valid_folder, batchsize, size=(h, w), class_mode='sparse')

    archi = BayesianClassifier(brickname='reference',
                               filters=[32, 64, 128],
                               kernels=[4, 5, 6],
                               strides=[1, 1, 1],
                               fc=[512, 512],
                               conv_activations=['relu', 'relu', 'relu'],
                               fc_activations=['relu', 'relu'],
                               end_activation=None,
                               output_channels=2)

    clf = BCLF(archi, height=h, width=w, colors=c, n_classes=2, learning_rate=lr, optimizer=opt, post_sampling=post_sampling)

    print(clf)

    trainlvals = []
    trainaccvals = []

    validlvals = []
    validaccvals = []

    accuracyplot = []

    i = 0
    lambdakl = 1.

    try:
        for e in range(epochs):

            print('EPOCH: ' + str(e + 1) + str('/') + str(epochs))

            print('TRAIN; kl coeff: ', str(lambdakl))

            trainprogressbar = tqdm(train_data, total=train_data.steps)

            for x, y in trainprogressbar:

                # lval, klval, accval = clf.fit(x, y)
                lval, accval = clf.fit(x, y, lambdakl)
                trainlvals.append(lval)
                trainaccvals.append(accval)
                metrics = [('loss', numpy.mean(trainlvals)), ('acc', numpy.mean(trainaccvals))]
                desc = monitor(metrics, 4)
                trainprogressbar.set_description(desc=desc, refresh=True)
                i += 1
                lambdakl = 2 ** (-i)

            trainlvals = []
            trainaccvals = []

            print('VALIDATE')

            validprogressbar = tqdm(valid_data, total=valid_data.steps)

            for x, y in validprogressbar:

                lval, accval = clf.validate(x, y, lambdakl)
                validlvals.append(lval)
                validaccvals.append(accval)
                metrics = [('logprob', numpy.mean(validlvals)), ('acc', numpy.mean(validaccvals))]
                desc = monitor(metrics, 4)
                validprogressbar.set_description(desc=desc, refresh=True)

            accuracyplot.append(numpy.mean(validaccvals))

            validlvals = []
            validaccvals = []

        clf.close(os.path.join(outfolder, 'model.ckpt'))

        with open(os.path.join(outfolder, 'accuracyplot.p'), 'wb') as f:
            pickle.dump(accuracyplot, f)

    # user can stop learning phase but the model will be saved
    except KeyboardInterrupt:

        print("user stopped iterations!!!")

        clf.close(os.path.join(outfolder, 'model.ckpt'))

        with open(os.path.join(outfolder, 'accuracyplot.p'), 'wb') as f:
            pickle.dump(accuracyplot, f)
