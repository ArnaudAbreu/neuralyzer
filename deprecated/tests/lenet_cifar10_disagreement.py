# coding: utf8
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle
from keras.datasets import cifar10
from keras.utils import to_categorical

# folders for ref and dep
# -----------------------
# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="folder to store disagreement dataset")

parser.add_argument("--basenet",
                    help="fully trained reference network")

parser.add_argument("--depnet",
                    help="fully trained dependency network")

args = parser.parse_args()
basenet = args.basenet
depnet = args.depnet
outfolder = args.outfolder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# load dataset (CIFAR10 here)
# ---------------------------
xtrain, ytrain, xtest, ytest = cifar10.load_data()
ytrain = to_categorical(ytrain, 10)
ytest = to_categorical(ytest, 10)


# load trained classifier (reference)
# -----------------------------------
# first, build architecture
refarchi = Classifier(brickname='lenet', dropouts=[0.5, 0.5], fcdropouts=[0.5])
# then, create the model
refclf = CLF(refarchi, model_path=basenet)
# predict with classifier
predref = refclf.predict(xtest)

# load trained classifier (dependency)
# ------------------------------------
# first, build architecture
deparchi = Classifier(brickname='explorer', dropouts=[0.5, 0.5], fcdropouts=[0.5])
# then, create the model
depclf = CLF(deparchi, model_path=depnet)
# predict with classifier
preddep = depclf.predict(xtest)

predrefclass = numpy.argmax(predref, axis=1)
preddepclass = numpy.argmax(preddep, axis=1)

preddiff = numpy.equal(predrefclass, preddepclass)

preddiff = numpy.logical_not(preddiff)

disagreement = xtest[preddiff == 1]

with open(os.path.join(outfolder, 'disagreement.p'), 'wb') as f:
    pickle.dump(disagreement, f)
