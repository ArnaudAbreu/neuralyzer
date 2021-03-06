# coding: utf8

from neuralyzer.data.wsi2pred import predict_slides
from neuralyzer.model import CLF, SiameseCLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle

# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--slidedir",
                    help="absolute path to the training directory")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--basenet",
                    help="path to 1 budget trained network")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

ref_path = args.basenet
outfolder = args.outfolder
slidedir = args.slidedir

h = 125
w = 125
c = 3

for experiment in range(100):

    experimentfolder = os.path.join(outfolder, 'exp_' + str(experiment))

    if not os.path.exists(experimentfolder):

        os.makedirs(experimentfolder)

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
                          output_channels=3)

    clf = CLF(refarchi, height=h, width=w, colors=c, n_classes=3, learning_rate=0.001, model_path=ref_path, optimizer="SGD")

    outputdir = os.path.join(experimentfolder, 'slides_prediction')

    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)

    # predict slides
    predict_slides(clf, slidedir, outputdir)

    clf.close()
