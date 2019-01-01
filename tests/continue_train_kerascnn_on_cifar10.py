# coding: utf8
from neuralyzer.data import Cifar10Generator as cifargen
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier, VGG
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle

# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=32,
                    help="int, size of batches")

parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--load",
                    help="previous model path")

args = parser.parse_args()
batchsize = args.batchsize
epochs = args.epochs
outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
modelpath = args.load


train_data = cifargen(batchsize)
valid_data = cifargen(batchsize, train=False)

h = train_data.x.shape[1]
w = train_data.x.shape[2]
if train_data.x.ndim > 3:
    c = train_data.x.shape[3]
else:
    c = 1

archi = VGG(brickname='kerasCNN',
            filters=[32, 64],
            kernels=[3, 3],
            strides=[1, 1],
            dropouts=[0.25, 0.25],
            thicknesses=[2, 2, 2],
            fc=[512],
            fcdropouts=[0.5],
            conv_activations=['relu', 'relu'],
            fc_activations=['relu'],
            end_activation='softmax',
            output_channels=10)

clf = CLF(archi, height=h, width=w, colors=c, learning_rate=0.01, optimizer='SGD', model_path=modelpath)

print(clf)

trainlvals = []
trainaccvals = []

validlvals = []
validaccvals = []

accplot = []

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

    accplot.append(numpy.mean(validaccvals))
    validlvals = []
    validaccvals = []

clf.close(os.path.join(outfolder, 'model.ckpt'))

with open(os.path.join(outfolder, 'accuracyplot.p'), 'wb') as f:
    pickle.dump(accplot, f)
