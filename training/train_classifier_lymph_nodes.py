# coding: utf8

from neuralyzer.data import FolderGenerator as foldgen
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--trainfolder",
                    help="absolute path to the training directory")

parser.add_argument("--validfolder",
                    help="absolute path to the validation directory")

parser.add_argument("--batchsize", type=int, default=10,
                    help="int, size of batches")

parser.add_argument("--patchsize", type=int, default=125,
                    help="int, size of patches in pixels")

parser.add_argument("--inputchannels", type=int, default=3,
                    help="int, number of input channels")

parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--opt", default="SGD",
                    help="optimizer")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

args = parser.parse_args()

train_folder = args.trainfolder
valid_folder = args.validfolder
batchsize = args.batchsize
epochs = args.epochs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
opt = args.opt
lr = args.lr
outfolder = args.outfolder

h = args.patchsize
w = args.patchsize
c = args.inputchannels

train_data = foldgen(train_folder, batchsize, size=(h, w))
valid_data = foldgen(valid_folder, batchsize, size=(h, w))


archi = Classifier(brickname='reference',
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

clf = CLF(archi, height=h, width=w, colors=c, n_classes=2, learning_rate=lr, optimizer=opt)

print(clf)

trainlvals = []
trainaccvals = []

validlvals = []
validaccvals = []

accuracyplot = []

for e in range(epochs):

    print('EPOCH: ' + str(e + 1) + str('/') + str(epochs))

    print('TRAIN')

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

    accuracyplot.append(numpy.mean(validaccvals))

    validlvals = []
    validaccvals = []

clf.close(os.path.join(outfolder, 'model.ckpt'))

with open(os.path.join(outfolder, 'accuracyplot.p'), 'wb') as f:
    pickle.dump(accuracyplot, f)
