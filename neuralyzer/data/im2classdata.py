# coding: utf8
from .data import *
import numpy
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.utils import print_summary, to_categorical
from keras.preprocessing.image import ImageDataGenerator
import os
from skimage.io import imread


class MNISTGenerator:

    def __init__(self, batchsize, seed=None, norm=True, train=True):

        if train:
            (self.x, self.y), _ = mnist.load_data()
        else:
            _, (self.x, self.y) = mnist.load_data()

        self.y = to_categorical(self.y, 10)

        self.steps = int(float(self.x.shape[0]) / batchsize)
        self.gen = NumpyGenerator(self.x, self.y, batchsize, seed=seed)
        self.norm = norm

    def __iter__(self):

        for step in range(self.steps):
            x, y = self.gen.next()

            if self.norm:

                x = [i.astype('float32') / 255. for i in x]

            yield x, y


class FashionGenerator:

    def __init__(self, batchsize, seed=None, norm=True, train=True):

        if train:
            (self.x, self.y), _ = fashion_mnist.load_data()
        else:
            _, (self.x, self.y) = fashion_mnist.load_data()

        self.y = to_categorical(self.y, 10)

        self.steps = int(float(self.x.shape[0]) / batchsize)
        self.gen = NumpyGenerator(self.x, self.y, batchsize, seed=seed)
        self.norm = norm

    def __iter__(self):

        for step in range(self.steps):
            x, y = self.gen.next()

            if self.norm:

                x = [i.astype('float32') / 255. for i in x]

            yield x, y


class Cifar10Generator:

    def __init__(self, batchsize, seed=None, norm=True, train=True):

        if train:
            (self.x, self.y), _ = cifar10.load_data()
        else:
            _, (self.x, self.y) = cifar10.load_data()

        self.y = to_categorical(self.y, 10)

        self.steps = int(float(self.x.shape[0]) / batchsize)
        self.gen = NumpyGenerator(self.x, self.y, batchsize, seed=seed)
        self.norm = norm

    def __iter__(self):

        for step in range(self.steps):
            x, y = self.gen.next()

            if self.norm:

                x = [i.astype('float32') / 255. for i in x]

            yield x, y


class FolderGenerator:

    def __init__(self, directory, batchsize, norm=True, size=(125, 125), class_mode='categorical'):

        # assume class have same number of samples
        # i.e. same number of images in each directory
        self.directory = directory
        imdir = os.path.join(directory, os.listdir(directory)[0])
        n_classes = len([d for d in os.listdir(directory) if d[0] != '.'])
        self.steps = 0
        for dirs in os.listdir(directory):
            self.steps += len(os.listdir(os.path.join(directory, dirs)))
        self.steps /= batchsize
        self.steps = int(self.steps)
        self.imgenobject = ImageDataGenerator()
        self.gen = self.imgenobject.flow_from_directory(directory, target_size=size, batch_size=batchsize, class_mode=class_mode)
        self.norm = norm

    def __iter__(self):

        for step in range(self.steps):

            x, y = self.gen.next()

            if self.norm:

                x = [i.astype('float32') / 255. for i in x]

            yield x, y
