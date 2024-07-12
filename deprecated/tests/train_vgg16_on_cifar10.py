# coding: utf8

from neuralyzer.data import MNISTGenerator as mnistgen
from neuralyzer.data import FashionGenerator as fashionmnistgen
from neuralyzer.data import Cifar10Generator as cifargen
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier, VGG
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor

# this file is a test for all functionalities of the deep ensemble

batchsize = 128

train_data = cifargen(batchsize)
valid_data = cifargen(batchsize, train=False)

h = train_data.x.shape[1]
w = train_data.x.shape[2]
if train_data.x.ndim > 3:
    c = train_data.x.shape[3]
else:
    c = 1

epochs = 100

archi = VGG(brickname='vgg16')

clf = CLF(archi, height=h, width=w, colors=c, learning_rate=0.01)

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
