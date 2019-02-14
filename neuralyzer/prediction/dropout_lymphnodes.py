# coding: utf8

from ..data.wsi2pred import predict_slides, dropout_predict_slides_from_labpathlist
from ..model import CLF
from ..archi import Classifier
from ..render import monitor
from tqdm import tqdm
import numpy
import argparse
import os
import pickle


def test(labpathfile, device, basenet, outfolder, predlevel, patchsize, patchinter, inputchannels):

    """
    A function to predict a list of slides given in a python pickled file
    computed during dataset creation.
    It's called test, because it runs on test directory with labeled wsi for
    comparison purpose.

    Arguments:
        - labpathfile: str, path to a file computed at dataset creation time.
        - device: str, '0' or '1', GPU card id.
        - basenet: str, path to a tensorflow model.ckpt.
        - outfolder: str, path to a folder to pickle prediction of each patch of each slide.
        - predlevel: int, wsi pyramid level to which perform prediction task.
        - patchsize: int, size of patches to provide to the network (side in pixels)
        depends on the size used for training, cannot differ from it...
        - patchinter: int, interval in pixels (given at predlevel) between two extracted patches.
        - inputchannels: int, number of channels in the image, usually 3.

    Returns:
        - Nothing, just store patch prediction for each slide in a pickled file.
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    h = patchsize
    w = patchsize
    c = inputchannels

    with open(labpathfile, 'rb') as f:
        labpathlist = pickle.load(f)

    refarchi = Classifier(brickname='reference',
                          filters=[32, 64, 128],
                          kernels=[4, 5, 6],
                          strides=[1, 1, 1],
                          dropouts=[0.1, 0.2, 0.25],
                          fc=[1024, 1024],
                          fcdropouts=[0.5, 0.5],
                          conv_activations=['relu', 'relu', 'relu'],
                          fc_activations=['relu', 'relu'],
                          end_activation='softmax',
                          output_channels=2)

    clf = CLF(refarchi, height=h, width=w, colors=c, n_classes=2, learning_rate=0.001, model_path=basenet, optimizer="SGD")

    outputdir = os.path.join(outfolder, 'slides_prediction')

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    # predict slides
    # predict_slides(clf, slidedir, outputdir, prediction_level=predlevel, n_classes=2)
    # predict_slides_from_dir(clf, slidedir, outputdir, predlevel, interval, w, h)
    dropout_predict_slides_from_labpathlist(clf, labpathlist, outputdir, predlevel, patchinter, w, h)

    clf.close()
