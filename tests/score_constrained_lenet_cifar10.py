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
import pickle

# this file is a test for all functionalities of the deep ensemble

logfolder = sys.argv[1]

refnetwork_folder = sys.argv[2]

coefficient = float(sys.argv[3])

lenet_number = int(sys.argv[4])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]

batchsize = 1

valid_data = cifargen(batchsize, train=False)

h = valid_data.x.shape[1]
w = valid_data.x.shape[2]
if valid_data.x.ndim > 3:
    c = valid_data.x.shape[3]
else:
    c = 1

ensemble_accuracies = []
ref_accuracy = 0.
dep_accuracies = []

for n in range(lenet_number):

    dirname = os.path.join(logfolder, 'dependency' + str(n))

    # first of all, evaluate reference alone
    if n == 0:

        archidep = Classifier(brickname='dependency' + str(n), dropouts=[0.5, 0.5], fcdropouts=[0.5])
        archiref = Classifier(brickname='reference', dropouts=[0.5, 0.5], fcdropouts=[0.5])

        clf = CLF(archiref, height=h, width=w, colors=c, learning_rate=0.0001, model_path=os.path.join(refnetwork_folder, 'model.ckpt'))

        print('SCORE REFERENCE')

        validprogressbar = tqdm(valid_data, total=valid_data.steps)

        accvals = []

        for x, y in validprogressbar:

            _, accval = clf.validate(x, y)
            accvals.append(accval)
            metrics = [('acc', numpy.mean(accvals))]
            desc = monitor(metrics, 4)
            validprogressbar.set_description(desc=desc, refresh=True)

        ref_accuracy = numpy.mean(accvals)

        clf.close()

        del clf, archidep, archiref

    # then evaluate dependency alone
    archidep = Classifier(brickname='dependency' + str(n), dropouts=[0.5, 0.5], fcdropouts=[0.5])
    archiref = Classifier(brickname='reference', dropouts=[0.5, 0.5], fcdropouts=[0.5])
    clf = CLF(archidep, height=h, width=w, colors=c, learning_rate=0.0001, model_path=os.path.join(dirname, 'model.ckpt'))

    print('SCORE DEPENDENCY')

    validprogressbar = tqdm(valid_data, total=valid_data.steps)

    accvals = []

    for x, y in validprogressbar:

        _, accval = clf.validate(x, y)
        accvals.append(accval)
        metrics = [('acc', numpy.mean(accvals))]
        desc = monitor(metrics, 4)
        validprogressbar.set_description(desc=desc, refresh=True)

    dep_accuracies.append(numpy.mean(accvals))

    clf.close()

    del clf, archidep, archiref

    # finally, evaluate siamese
    archidep = Classifier(brickname='dependency' + str(n), dropouts=[0.5, 0.5], fcdropouts=[0.5])
    archiref = Classifier(brickname='reference', dropouts=[0.5, 0.5], fcdropouts=[0.5])
    clf = SiameseCLF(archiref, archidep, coeff=coefficient, height=h, width=w, colors=c, learning_rate=0.0001, ref_path=os.path.join(refnetwork_folder, 'model.ckpt'), dep_path=os.path.join(dirname, 'model.ckpt'))

    print('SCORE ENSEMBLE')

    validprogressbar = tqdm(valid_data, total=valid_data.steps)
    accvals = []

    for x, y in validprogressbar:

        accval = clf.score_ensemble(x, y)
        accvals.append(accval)
        metrics = [('acc', numpy.mean(accvals))]
        desc = monitor(metrics, 4)
        validprogressbar.set_description(desc=desc, refresh=True)

    ensemble_accuracies.append(numpy.mean(accvals))

    clf.close()

print('\nglobal dependency accuracy: ' + str(numpy.mean(dep_accuracies)) + '\n')
print('\nglobal ensemble accuracy: ' + str(numpy.mean(ensemble_accuracies)) + '\n')
print('\nreference accuracy: ' + str(ref_accuracy))

with open(os.path.join(logfolder, 'accuracy_histos.p'), 'wb') as f:
    pickle.dump({'dependency': dep_accuracies, 'ensemble': ensemble_accuracies, 'reference': ref_accuracy}, f)
