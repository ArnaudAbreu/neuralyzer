# coding: utf8
from neuralyzer.data import patch
from openslide import OpenSlide

import argparse
import os
from tqdm import tqdm
import numpy
import pickle

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

parser.add_argument("--classname", type=str, help="class name, i.e. folder name.")

args = parser.parse_args()


slidenames = [f for f in os.listdir(args.infolder) if f[0] != '.' and '.mrxs' in f]

stopidx = int(args.trainratio * len(slidenames))

slidenames = numpy.array(slidenames)

numpy.random.shuffle(slidenames)

labpathlistfolder = os.path.join(args.outfolder, '..')
labpathlist = []

trainfolder = os.path.join(args.outfolder, 'Train')
trainfolder = os.path.join(trainfolder, args.classname)

testfolder = os.path.join(args.outfolder, 'Test')
testfolder = os.path.join(testfolder, args.classname)

if not os.path.exists(trainfolder):
    os.makedirs(trainfolder)

if not os.path.exists(testfolder):
    os.makedirs(testfolder)


print('start train patchification:')

for slidename in tqdm(slidenames[0:stopidx]):

    slide = OpenSlide(os.path.join(args.infolder, slidename))
    prefix = os.path.splitext(slidename)[0]
    prefix = os.path.join(trainfolder, prefix)

    patch.patchify(slide, args.level, args.interval, args.sizex, args.sizey, prefix)


print('start test patchification:')

for slidename in tqdm(slidenames[stopidx::]):

    labpathlist.append((os.path.join(args.infolder, slidename), args.classname))
    slide = OpenSlide(os.path.join(args.infolder, slidename))
    prefix = os.path.splitext(slidename)[0]
    prefix = os.path.join(testfolder, prefix)

    patch.patchify(slide, args.level, args.interval, args.sizex, args.sizey, prefix)


with open(os.path.join(labpathlistfolder, 'labpathlist.p'), 'wb') as f:
    pickle.dump(labpathlist, f)
