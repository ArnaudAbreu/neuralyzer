# coding: utf8

from neuralyzer.data.wsi2pred import dropout_predict_slides_from_dir_with_tree, dropout_predict_slides_from_labpathlist_with_tree
from neuralyzer.model import CCLF
from neuralyzer.archi import Classifier
from neuralyzer.render import monitor
from tqdm import tqdm
import numpy
import argparse
import os
import pickle

parser = argparse.ArgumentParser()

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--inputdir", type=str, help="path to slide folder.")

parser.add_argument("--labpathdir", default="None", type=str, help="path to a folder containing labpath file.")

parser.add_argument("--outputdir", type=str, help="path to output folder.")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

parentdir = '/data/DeepLearning/ABREU_Arnaud/SYRYKH_Charlotte'

modeldirs = {k: os.path.join(os.path.join(parentdir, 'DropClassifP' + str(k)), 'Model') for k in range(4, 8)}

# outputdir = '/data/DeepLearning/ABREU_Arnaud/SYRYKH_Charlotte/Preds4to7TreeFashion'

# dirnames = ['/data/DeepLearning/ABREU_Arnaud/SYRYKH_Charlotte/LF3', '/data/DeepLearning/ABREU_Arnaud/SYRYKH_Charlotte/HF3']


class ModelCollection:

    def __init__(self, modeldirs):

        self.modeldirs = modeldirs
        self.patchsize = 125

    def load_level(self, level):

        basenet = os.path.join(self.modeldirs[level], 'model.ckpt')

        self.archi = Classifier(brickname='reference',
                                filters=[32, 64, 128],
                                kernels=[4, 5, 6],
                                strides=[1, 1, 1],
                                dropouts=[0., 0., 0.],
                                fc=[1024, 1024],
                                fcdropouts=[0.5, 0.5],
                                conv_activations=['relu', 'relu', 'relu'],
                                fc_activations=['relu', 'relu'],
                                end_activation='softmax',
                                output_channels=2)

        self.clf = CCLF(self.archi, height=125, width=125, colors=3, n_classes=2, learning_rate=0.001, model_path=basenet, optimizer="SGD", sampling=100)

    def close_level(self):

        self.clf.close()

        del self.clf, self.archi


my_models = ModelCollection(modeldirs)

dropout_predict_slides_from_dir_with_tree(my_models, args.inputdir, args.outputdir, 7, 4)
