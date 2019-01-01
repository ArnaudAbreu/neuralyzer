# coding: utf8
from neuralyzer.data import HEVGenerator as hevgen
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

parser.add_argument("--trainfolder",
                    help="absolute path to the training directory")

parser.add_argument("--validfolder",
                    help="absolute path to the validation directory")

parser.add_argument("--batchsize", type=int, default=10,
                    help="int, size of batches")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--basenet",
                    help="path to 1 budget trained network")

parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate for training")

parser.add_argument("--opt", default="SGD",
                    help="optimizer")

parser.add_argument('--epochs', type=int, default=300,
                    help="number of epochs to train networks")

args = parser.parse_args()
batchsize = args.batchsize
outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
learning_rate = args.lr
ref_path = args.basenet
opt = args.opt
train_folder = args.trainfolder
valid_folder = args.validfolder
epochs = args.epochs

train_data = hevgen(train_folder, batchsize)
valid_data = hevgen(valid_folder, batchsize)

h = 125
w = 125
c = 3

for experiment in range(100):

    experimentfolder = os.path.join(outfolder, 'const_' + str(experiment))
    if not os.path.exists(experimentfolder):
        os.makedirs(experimentfolder)

    for k in [0., 0.2, 0.5]:

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

        constarchi = Classifier(brickname='explorer',
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

        clf = SiameseCLF(refarchi, constarchi, coeff=k, height=h, width=w, colors=c, n_classes=3, learning_rate=learning_rate, ref_path=ref_path, optimizer=opt)

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

        for epoch in range(epochs):

            print('EPOCH: ' + str(epoch) + str('/') + str(epochs))

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

            validepaccvals = []
            validrefaccvals = []
            validensaccvals = []

        clf.close(os.path.join(kvarfolder, 'model.ckpt'))

        with open(os.path.join(kvarfolder, 'accuracyplot.p'), 'wb') as f:
            pickle.dump({'dependency': depaccplot, 'ensemble': meanaccplot}, f)
