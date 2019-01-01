# coding: utf8

from neuralyzer.data import MNISTGenerator as mnistgen
from neuralyzer.data import FashionGenerator as fashionmnistgen
from neuralyzer.data import Cifar10Generator as cifargen
from neuralyzer.model import CLF, SiameseCLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import sys
import os

import argparse
import os
import pickle

# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=32,
                    help="int, size of batches")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--basenet",
                    help="path to 075 budget trained network")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate for training")

args = parser.parse_args()
batchsize = args.batchsize
outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
learning_rate = args.lr
explorer_path = args.basenet

train_data = cifargen(batchsize)
valid_data = cifargen(batchsize, train=False)

h = train_data.x.shape[1]
w = train_data.x.shape[2]
if train_data.x.ndim > 3:
    c = train_data.x.shape[3]
else:
    c = 1

for k in numpy.arange(start=0., step=0.1, stop=0.65):

    new_folder = str(k).split('.')
    corrected_new_folder = ''
    for fold in range(len(new_folder)):
        if fold == 0:
            corrected_new_folder += (new_folder[fold] + '_')
        else:
            if len(new_folder[fold]) > 1:
                corrected_new_folder += new_folder[fold][0]
            else:
                corrected_new_folder += new_folder[fold]

    logfolder = os.path.join(constfolder, corrected_new_folder)

    if not os.path.isdir(logfolder):
        os.makedirs(logfolder)

    for n in range(lenet_number):

        epochs = 20

        archidep = Classifier(brickname='dependency' + str(n), dropouts=[0.5, 0.5], fcdropouts=[0.5])
        archiref = Classifier(brickname='reference', dropouts=[0.5, 0.5], fcdropouts=[0.5])

        clf = SiameseCLF(archiref, archidep, coeff=k, height=h, width=w, colors=c, learning_rate=0.01, ref_path=os.path.join(refnetwork_folder, 'model.ckpt'))

        print(clf)

        trainlvals = []
        trainaccvals = []

        validlvals = []
        validaccvals = []

        e = 0

        while True:

            print('EPOCH: ' + str(e + 1) + str('/') + str(epochs))

            print('TRAIN (k = ' + str(k) + ')')

            trainprogressbar = tqdm(train_data, total=train_data.steps)

            for x, y in trainprogressbar:

                lval, accval = clf.fit(x, y)
                trainlvals.append(lval)
                trainaccvals.append(accval)
                metrics = [('loss', numpy.mean(trainlvals)), ('acc', numpy.mean(trainaccvals))]
                desc = monitor(metrics, 4)
                trainprogressbar.set_description(desc=desc, refresh=True)

            trainlvals = []
            trainaccvals = []

            print('VALIDATE')

            validprogressbar = tqdm(valid_data, total=valid_data.steps)

            for x, y in validprogressbar:

                lval, accval = clf.validate(x, y)
                validlvals.append(lval)
                validaccvals.append(accval)
                metrics = [('loss', numpy.mean(validlvals)), ('acc', numpy.mean(validaccvals))]
                desc = monitor(metrics, 4)
                validprogressbar.set_description(desc=desc, refresh=True)
            if numpy.mean(validaccvals) >= 0.6542:
                print('REACHED ACCURACY OF REFERENCE\n')
                break

            e += 1

            validlvals = []
            validaccvals = []

        dirname = os.path.join(logfolder, 'dependency' + str(n))

        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            clf.close(os.path.join(dirname, 'model.ckpt'))
