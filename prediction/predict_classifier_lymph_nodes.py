# coding: utf8

from neuralyzer.data.wsi2pred import predict_slides, predict_slidesV2
from neuralyzer.model import CLF
from neuralyzer.archi import Classifier
from tqdm import tqdm
import numpy
from neuralyzer.render import monitor
import argparse
import os
import pickle


parser = argparse.ArgumentParser()

parser.add_argument("--slidedir",
                    help="absolute path to the directory to predict")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--basenet",
                    help="path to 1 budget trained network")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--predlevel", type=int, default=5,
                    help="pyramid level for patch prediction")

parser.add_argument("--patchsize", type=int, default=125,
                    help="int, size of patches in pixels")

parser.add_argument("--patchinter", type=int, default=125,
                    help="int, interval between patches in pixels")

parser.add_argument("--inputchannels", type=int, default=3,
                    help="int, number of input channels")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

ref_path = args.basenet
outfolder = args.outfolder
slidedir = args.slidedir
patch_level = args.predlevel
interval = args.patchinter

h = args.patchsize
w = args.patchsize
c = args.inputchannels


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
                      output_channels=2)

clf = CLF(refarchi, height=h, width=w, colors=c, n_classes=2, learning_rate=0.001, model_path=ref_path, optimizer="SGD")

outputdir = os.path.join(outfolder, 'slides_prediction')

if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

# predict slides
# predict_slides(clf, slidedir, outputdir, prediction_level=predlevel, n_classes=2)
predict_slidesV2(clf, slidedir, outputdir, patch_level, interval, w, h)

clf.close()
