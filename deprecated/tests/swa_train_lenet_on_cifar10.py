# coding: utf8

from neuralyzer.data import MNISTGenerator as mnistgen
from neuralyzer.data import FashionGenerator as fashionmnistgen
from neuralyzer.data import Cifar10Generator as cifargen
from neuralyzer.model import CLF, SWACLF
from neuralyzer.archi import Classifier, ClassifierAccumulator
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

for experiment in range(100):

    experimentfolder = os.path.join(outfolder, 'swa_' + str(experiment))
    if not os.path.exists(experimentfolder):
        os.makedirs(experimentfolder)

    for epochs in [100, 150, 200]:

        base_archi = Classifier(brickname='lenet', dropouts=[0.5, 0.5], fcdropouts=[0.5])
        swa_archi = ClassifierAccumulator(brickname='explorer', dropouts=[0.5, 0.5], fcdropouts=[0.5])

        clf = SWACLF(base_archi, swa_archi, height=h, width=w, colors=c, learning_rate=learning_rate, explorer_path=explorer_path, optimizer='SGD')

        budgetfolder = os.path.join(experimentfolder, str(int((150 + epochs) * 0.5)) + 'Budget')

        if not os.path.exists(budgetfolder):
            os.makedirs(budgetfolder)

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

            if e == 0:
                clf.init_acc()
            else:
                clf.accumulate(e + 1)

            print('VALIDATE')

            validprogressbar = tqdm(valid_data, total=valid_data.steps)

            for x, y in validprogressbar:

                lval, accval = clf.validate(x, y)
                validlvals.append(lval)
                validaccvals.append(accval)
                metrics = [('acc', numpy.mean(validlvals)), ('accswa', numpy.mean(validaccvals))]
                desc = monitor(metrics, 4)
                validprogressbar.set_description(desc=desc, refresh=True)

            accplot.append(numpy.mean(validaccvals))
            validlvals = []
            validaccvals = []

        clf.close(os.path.join(budgetfolder, 'model.ckpt'))

        with open(os.path.join(budgetfolder, 'accuracyplot.p'), 'wb') as f:
            pickle.dump(accplot, f)
