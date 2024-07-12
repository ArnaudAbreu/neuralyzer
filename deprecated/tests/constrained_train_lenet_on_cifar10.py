# coding: utf8

from neuralyzer.data import MNISTGenerator as mnistgen
from neuralyzer.data import FashionGenerator as fashionmnistgen
from neuralyzer.data import Cifar10Generator as cifargen
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

parser.add_argument("--tol", type=float, default=0.0,
                    help="tolerance for accuracy drop")

args = parser.parse_args()
batchsize = args.batchsize
outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
learning_rate = args.lr
ref_path = args.basenet
tolerance = args.tol

train_data = cifargen(batchsize)
valid_data = cifargen(batchsize, train=False)

h = train_data.x.shape[1]
w = train_data.x.shape[2]
if train_data.x.ndim > 3:
    c = train_data.x.shape[3]
else:
    c = 1

for experiment in range(100):

    experimentfolder = os.path.join(outfolder, 'const_' + str(experiment))
    if not os.path.exists(experimentfolder):
        os.makedirs(experimentfolder)

    for k in numpy.arange(start=0., step=0.1, stop=0.65):

        refarchi = Classifier(brickname='lenet', dropouts=[0.5, 0.5], fcdropouts=[0.5])
        constarchi = Classifier(brickname='explorer', dropouts=[0.5, 0.5], fcdropouts=[0.5])

        clf = SiameseCLF(refarchi, constarchi, coeff=k, height=h, width=w, colors=c, learning_rate=learning_rate, ref_path=ref_path, optimizer='SGD')

        kvarfolder = os.path.join(experimentfolder, str((int(k * 10))))

        if not os.path.exists(kvarfolder):
            os.makedirs(kvarfolder)

        # print(clf)

        trainlvals = []
        trainaccvals = []

        validepaccvals = []
        validrefaccvals = []
        validensaccvals = []

        depaccplot = []
        meanaccplot = []

        e = 0
        stab = 0
        prevacc = 0.

        while True:

            print('EPOCH: ' + str(e + 1) + str('/UNKNOWN'))

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

                refaccval, depaccval, ensaccval = clf.score_ensemble(x, y)
                validrefaccvals.append(refaccval)
                validepaccvals.append(depaccval)
                validensaccvals.append(ensaccval)
                metrics = [('acc', numpy.mean(validepaccvals)), ('accmean', numpy.mean(validensaccvals))]
                desc = monitor(metrics, 4)
                validprogressbar.set_description(desc=desc, refresh=True)

            meanaccplot.append(numpy.mean(validensaccvals))
            depaccplot.append(numpy.mean(validepaccvals))

            if numpy.mean(validepaccvals) >= (numpy.mean(validrefaccvals) - tolerance):
                print('REACHED ACCURACY OF REFERENCE\n')
                break
            elif numpy.mean(validepaccvals) - prevacc < 0. and e > 200:
                stab += 1
                if stab > 20:
                    print('REACHED STABILIZATION\n')
                    break
            else:
                print('accuracy: ', numpy.mean(validepaccvals), ' < ', 'ref accuracy: ', numpy.mean(validrefaccvals) - tolerance)

            e += 1
            prevacc = numpy.mean(validepaccvals)

            validepaccvals = []
            validrefaccvals = []
            validensaccvals = []

        clf.close(os.path.join(kvarfolder, 'model.ckpt'))

        with open(os.path.join(kvarfolder, 'accuracyplot.p'), 'wb') as f:
            pickle.dump({'dependency': depaccplot, 'ensemble': meanaccplot}, f)
