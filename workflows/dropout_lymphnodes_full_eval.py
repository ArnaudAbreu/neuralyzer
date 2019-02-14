# coding: utf8
from neuralyzer.data.lymphnodes import generate
from neuralyzer.training.lymphnodes import train
from neuralyzer.prediction.dropout_lymphnodes import test
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--level", type=int, default=5,
                    help="int, pyramid level, sample resolution.")

parser.add_argument("--interval", type=int, default=125,
                    help="int, interval between two samples, at level=--level.")

parser.add_argument("--sizex", type=int, default=125,
                    help="int, size of sample on x axis, at level=--level.")

parser.add_argument("--sizey", type=int, default=125,
                    help="int, size of sample on y axis, at level=--level.")

parser.add_argument("--trainratio", type=float, default=0.66,
                    help="float, percentage of data to put in training set.")

parser.add_argument("--infolder", type=str, help="path to slide folder.")

parser.add_argument("--outfolder", type=str, help="path to outfolder.")

parser.add_argument("--batchsize", type=int, default=10,
                    help="int, size of batches")

parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--opt", default="SGD",
                    help="optimizer")

args = parser.parse_args()

outpatchfolder = os.path.join(args.outfolder, 'Data')
train_folder = os.path.join(outpatchfolder, 'Train')
valid_folder = os.path.join(outpatchfolder, 'Test')
network_folder = os.path.join(args.outfolder, 'Model')
labpathfile = os.path.join(args.outfolder, 'labpathlist.p')
network_file = os.path.join(network_folder, 'model.ckpt')
eval_folder = os.path.join(args.outfolder, 'Evaluation')


generate(args.infolder, outpatchfolder, args.trainratio, args.level, args.sizex, args.sizey, args.interval)
train(train_folder, valid_folder, args.batchsize, args.sizex, 3, args.epochs, args.device, args.lr, args.opt, network_folder)
test(labpathfile, args.device, network_file, eval_folder, args.level, args.sizex, args.interval, 3)
