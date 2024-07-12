# coding: utf8
from .data import *
import numpy
from keras.datasets import mnist


class Im2ImGenerator:

    def __init__(self, inputimdir, outputimdir, batchsize, iterations, seed=None):

        self.inputgen = Generator(inputimdir, batchsize, seed)
        self.outputgen = Generator(outputimdir, batchsize, seed)
        self.steps = iterations

    def __iter__(self):

        for step in range(self.steps):

            x = self.inputgen.next()
            y = self.outputgen.next()

            yield x, y


class PatchFromNumpyGenerator:

    def __init__(self, xarray, yarray, npatches, size):

        self.x = xarray
        self.y = yarray
        self.npatches = npatches
        self.size = size

    def get_epoch_data(self, seed=None):

        if seed is None:
            usedSeed = numpy.random.randint(2**32 - 1)
        else:
            usedSeed = seed

        numpy.random.seed(seed)

        res_X = []
        res_Y = []

        for X, Y in zip(self.x, self.y):

            xs = image.extract_patches_2d(X, (self.size, self.size), max_patches=self.npatches, random_state=usedSeed)
            ys = image.extract_patches_2d(Y, (self.size, self.size), max_patches=self.npatches, random_state=usedSeed)

            for x, y in zip(xs, ys):

                # if no pixel outside
                if (y < 0.5).sum() == 0:

                    res_X.append(x)
                    res_Y.append(y[:, :] - 1)

        res_X = numpy.array(res_X)
        res_Y = numpy.array(res_Y)

        perm = numpy.random.permutation(len(res_X))

        return res_X[perm], res_Y[perm]
