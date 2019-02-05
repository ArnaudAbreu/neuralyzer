# coding: utf8
from .patch import patchify
from openslide import OpenSlide

import argparse
import os
from tqdm import tqdm
import numpy
import pickle


def generate_class(infolder, outfolder, classname, trainratio, level, sizex, sizey, interval, labpathlist):

    """
    A function to compute train and test images for a given class and from a
    folder of mrxs slides. It stores the paths and labels of test images in a
    list for later testing procedure.

    Arguments:
        - infolder: str, path to mrxs files of a given class <=> diagnosis.
        - outfolder: str, path to empty or not existing folder to store separately
        test and training samples.
        - classname: str, name of the class, used for storage folders.
        - trainratio: float, ratio of images to use for training.
        - level: int, wsi pyramidal level to extract patches.
        - sizex: int, size of patches on x axis, in pixels at extraction level.
        - sizey: int, size of patches on y axis, in pixels at extraction level.
        - interval: int, distance between patch centers, in pixels at extraction level.
        - labpathlist: list of tuples, (mrxsfilepath (str), classname (str)).

    Returns:
        - Nothing, just generate patches in a folder and append labpathlist with
        test files.
    """

    slidenames = [f for f in os.listdir(infolder) if f[0] != '.' and '.mrxs' in f]

    stopidx = int(trainratio * len(slidenames))

    slidenames = numpy.array(slidenames)

    numpy.random.shuffle(slidenames)

    trainfolder = os.path.join(outfolder, 'Train')
    trainfolder = os.path.join(trainfolder, classname)

    testfolder = os.path.join(outfolder, 'Test')
    testfolder = os.path.join(testfolder, classname)

    if not os.path.exists(trainfolder):
        os.makedirs(trainfolder)

    if not os.path.exists(testfolder):
        os.makedirs(testfolder)

    print('start train patchification:')

    for slidename in tqdm(slidenames[0:stopidx]):

        slide = OpenSlide(os.path.join(infolder, slidename))
        prefix = os.path.splitext(slidename)[0]
        prefix = os.path.join(trainfolder, prefix)

        patch.patchify(slide, level, interval, sizex, sizey, prefix)

    print('start test patchification:')

    for slidename in tqdm(slidenames[stopidx::]):

        labpathlist.append((os.path.join(infolder, slidename), classname))
        slide = OpenSlide(os.path.join(infolder, slidename))
        prefix = os.path.splitext(slidename)[0]
        prefix = os.path.join(testfolder, prefix)

        patch.patchify(slide, level, interval, sizex, sizey, prefix)


def generate(infolder, outfolder, trainratio, level, sizex, sizey, interval):

    """
    A function to compute train and test images for 'HF' and 'LF' classes for
    lymphnodes diagnosis from a folder containing 'LF' and 'HF' subfolders of
    mrxs files. It extract patches and store test mrxs paths with associated
    labels for later testing procedure in a 'labpathlist.p' file.

    Arguments:
        - infolder: str, path to a folder containing subfolders 'HF' and 'LF'
        folders.
        - outfolder: str, path to a folder where to store patches.
        - trainratio: float, ratio of slides to use for training.
        - level: int, wsi pyramidal level to extract patches.
        - sizex: int, size of patches on x axis, in pixels at extraction level.
        - sizey: int, size of patches on y axis, in pixels at extraction level.
        - interval: int, distance between patch centers, in pixels at extraction level.

    Returns:
        - Nothing, just generate patches in appropriate folders for later training-
        testing generation.
    """

    hf_infolder = os.path.join(infolder, 'HF')
    lf_infolder = os.path.join(infolder, 'LF')
    labpathlistfolder = os.path.join(args.outfolder, '..')
    labpathlistfile = os.path.join(labpathlistfolder, 'labpathlist.p')

    labpathlist = []

    print('HF patches generation:')
    print('#' * 20)
    generate_class(hf_infolder,
                   outfolder,
                   'HF',
                   trainratio,
                   level,
                   sizex,
                   sizey,
                   interval,
                   labpathlist)

    print('LF patches generation:')
    print('#' * 20)
    generate_class(lf_infolder,
                   outfolder,
                   'LF',
                   trainratio,
                   level,
                   sizex,
                   sizey,
                   interval,
                   labpathlist)

    with open(labpathlistfile, 'wb') as f:
        pickle.dump(labpathlist, f)
