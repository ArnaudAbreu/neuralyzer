# coding: utf8

from neuralyzer.data import PatchFromNumpyGenerator
from neuralyzer.model import VnetDist
from neuralyzer.archi import Vnet
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle


# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=28,
                    help="int, size of batches")

parser.add_argument("--epochs", type=int, default=300,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate for training")

parser.add_argument("--indir",
                    help="data directory")

parser.add_argument("--size", type=int, default=128,
                    help="patch size")

parser.add_argument("--ncat", type=int, default=24,
                    help="number of categories")

args = parser.parse_args()
batchsize = args.batchsize
epochs = args.epochs
outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
learning_rate = args.lr


x_train_data_filename = os.path.join(args.indir, "train/X/images.npy")
y_train_data_filename = os.path.join(args.indir, "train/Y/labels.npy")
x_test_data_filename = os.path.join(args.indir, "test/X/images.npy")
y_test_data_filename = os.path.join(args.indir, "test/Y/labels.npy")

Xtr = numpy.load(x_train_data_filename)
Ytr = numpy.load(y_train_data_filename)
Xte = numpy.load(x_test_data_filename)
Yte = numpy.load(y_test_data_filename)

training_data_generator = PatchFromNumpyGenerator(Xtr, Ytr, 16, 128)
test_data_generator = PatchFromNumpyGenerator(Xte, Yte, 16, 128)

# epochsize = 16 * Xtr.shape[0]
# n_batches = int(epochsize / batchsize)


def to_categorical_pacth_overlaping(y, nb_cat):

    y_cat = numpy.zeros([y.shape[0], args.size, args.size, nb_cat], dtype=numpy.int32)
    y_cat[:, :, :, 0] = (y == 0).astype(numpy.int32)

    for i in range(1, nb_cat):
        y_cat[:, :, :, i] = (y >= i).astype(numpy.int32)

    return y_cat


h = 128
w = 128
c = 3
ncat = 24

archi = Vnet(n_classes=ncat)

vnet = VnetDist(archi, height=h, width=w, colors=c, n_classes=ncat)

print(vnet)

# only loss to monitor
trainlvals = []
validlvals = []
lossplot = []

try:

    for e in range(epochs):

        print('EPOCH: ' + str(e + 1) + str('/') + str(epochs))

        print('TRAIN')

        xtrainepoch, ytrainepoch_ = training_data_generator.get_epoch_data()
        xtestepoch, ytestepoch_ = test_data_generator.get_epoch_data()
        #
        ytrainepoch = to_categorical_pacth_overlaping(ytrainepoch_, ncat)
        ytestepoch = to_categorical_pacth_overlaping(ytestepoch_, ncat)

        epochsize = xtrainepoch.shape[0]
        n_batches = int(epochsize / batchsize)

        trainprogressbar = tqdm(range(n_batches))

        for batchid in trainprogressbar:

            lval = vnet.fit(xtrainepoch[batchid * batchsize:(batchid + 1) * batchsize], ytrainepoch[batchid * batchsize:(batchid + 1) * batchsize])
            trainlvals.append(lval)
            metrics = [('loss', numpy.mean(trainlvals))]
            desc = monitor(metrics, 4)
            trainprogressbar.set_description(desc=desc, refresh=True)

        trainlvals = []

        print('VALIDATE')

        validprogressbar = tqdm(range(xtestepoch.shape[0]))

        for batchid in validprogressbar:

            lval = vnet.validate([xtestepoch[batchid]], [ytestepoch[batchid]])
            validlvals.append(lval)
            metrics = [('loss', numpy.mean(validlvals))]
            desc = monitor(metrics, 4)
            validprogressbar.set_description(desc=desc, refresh=True)

        lossplot.append(numpy.mean(validlvals))
        validlvals = []
        validaccvals = []

except KeyboardInterrupt:
    print('user interrupted training procedure!!!')


vnet.close(os.path.join(outfolder, 'model.ckpt'))

with open(os.path.join(outfolder, 'loss.p'), 'wb') as f:
    pickle.dump(lossplot, f)
