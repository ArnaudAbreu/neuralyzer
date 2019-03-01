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
from skimage.util import view_as_windows
from skimage.io import imread


# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate for training")

parser.add_argument("--infile",
                    help="image file")

parser.add_argument("--size", type=int, default=128,
                    help="patch size")

parser.add_argument("--basenet",
                    help="path to 1 budget trained network")

args = parser.parse_args()
# outfolder = args.outfolder
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
learning_rate = args.lr

h = 128
w = 128
c = 3
ncat = 12
step = 64

winshape = (h, w, c)

input_image = imread(args.infile)

patches = view_as_windows(input_image, winshape, step=step)

# number of word lines
plines = patches.shape[0]
# number of word columns
pcols = patches.shape[1]

patches = patches.reshape(plines * pcols, h, w, c)

archi = Vnet(n_classes=ncat)

vnet = VnetDist(archi, height=h, width=w, colors=c, n_classes=ncat, model_path=args.basenet)

print(vnet)

predictions = vnet.predict(patches)

pred = predictions[0]

print('output prediction shape: ', pred.shape)
print('output prediction dtype: ', pred.dtype)
print('output prediction vals: ', numpy.unique(pred))

predictionmap = numpy.zeros((input_image.shape[0], input_image.shape[1], ncat))

for predidx in range(predictions[0].shape[0]):

    line = int(predidx / plines)
    col = int(predidx % plines)

    predictionmap[line * step:line * step + h, col * step:col * step + w] += predictions[0][predidx]

with open(os.path.splitext(args.infile)[0] + '_nuclei_prediction.p', 'wb') as f:
    pickle.dump(predictionmap, f)
