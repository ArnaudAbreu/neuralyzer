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
