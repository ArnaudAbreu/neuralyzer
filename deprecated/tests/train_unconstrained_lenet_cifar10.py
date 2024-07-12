# coding: utf8

from neuralyzer.data import MNISTGenerator as mnistgen
from neuralyzer.data import FashionGenerator as fashionmnistgen
from neuralyzer.data import Cifar10Generator as cifargen
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import sys
import os

# this file is a test for all functionalities of the deep ensemble
logfolder = sys.argv[1]

lenet_number = int(sys.argv[2])

batchsize = 32

train_data = cifargen(batchsize)
valid_data = cifargen(batchsize, train=False)

print('Y shape: ', train_data.y.shape)

h = train_data.x.shape[1]
w = train_data.x.shape[2]
if train_data.x.ndim > 3:
    c = train_data.x.shape[3]
else:
    c = 1

for n in range(lenet_number):

    epochs = 10

    archi = Classifier(brickname='dependency' + str(n), dropouts=[0.5, 0.5], fcdropouts=[0.5])

    clf = CLF(archi, height=h, width=w, colors=c, learning_rate=0.0001)

    print(clf)

    trainlvals = []
    trainaccvals = []

    validlvals = []
    validaccvals = []

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

        validlvals = []
        validaccvals = []

    dirname = os.path.join(logfolder, 'dependency' + str(n))

    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        clf.close(os.path.join(dirname, 'model.ckpt'))
