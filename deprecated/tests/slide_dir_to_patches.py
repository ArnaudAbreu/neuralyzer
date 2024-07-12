# coding: utf8
from neuralyzer.data import patch
from openslide import OpenSlide

import argparse
import os
from tqdm import tqdm

# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--level", type=int, default=5,
                    help="int, pyramid level, sample resolution.")

parser.add_argument("--interval", type=int, default=125,
                    help="int, interval between two samples, at level=--level.")

parser.add_argument("--sizex", type=int, default=125,
                    help="int, size of sample on x axis, at level=--level.")

parser.add_argument("--sizey", type=int, default=125,
                    help="int, size of sample on y axis, at level=--level.")

parser.add_argument("--infolder", type=str, help="path to slide folder.")

parser.add_argument("--outfolder", type=str, help="path to outfolder.")

args = parser.parse_args()


slidenames = [f for f in os.listdir(args.infolder) if f[0] != '.' and '.mrxs' in f]

print('start patchification:')

for slidename in tqdm(slidenames):

    slide = OpenSlide(os.path.join(args.infolder, slidename))
    prefix = os.path.splitext(slidename)[0]
    prefix = os.path.join(args.outfolder, prefix)

    patch.patchify(slide, args.level, args.interval, args.sizex, args.sizey, prefix)
