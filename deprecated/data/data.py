# coding: utf8
import numpy
import os
from skimage.io import imread
from sklearn.feature_extraction import image


class FolderGenerator:

    def __init__(self, imdir, batchsize, seed=None):

        if seed is not None:
            numpy.random.seed(seed)
        else:
            numpy.random.seed(numpy.random.randint(100))
        self.file_list = [os.path.join(imdir, f) for f in os.listdir(imdir) if f[0] != '.' and os.path.splitext(f)[-1] in ['.png', '.jpg', '.tif', '.tiff', '.bmp']]
        numpy.random.shuffle(self.file_list)
        self.batchsize = batchsize
        self.k = 0

    def next(self):

        image_batch = []

        for b in range(self.batchsize):

            image_batch.append(imread(self.file_list[self.k]))
            if self.k < len(self.file_list) - 1:
                self.k += 1
            else:
                self.k = 0

        return image_batch


class NumpyGenerator:

    def __init__(self, xarray, yarray, batchsize, seed=None):

        if seed is not None:
            numpy.random.seed(seed)
        else:
            numpy.random.seed(numpy.random.randint(100))

        indices = numpy.arange(yarray.shape[0])

        numpy.random.shuffle(indices)
        self.x = xarray[indices]
        self.y = yarray[indices]
        self.batchsize = batchsize
        self.k = 0

    def next(self):

        x_batch = []
        y_batch = []

        for b in range(self.batchsize):

            if self.x.ndim < 4:
                x_batch.append(numpy.expand_dims(self.x[self.k], axis=-1))
            else:
                x_batch.append(self.x[self.k])
            y_batch.append(self.y[self.k])

            if self.k < self.y.shape[0] - 1:
                self.k += 1
            else:
                self.k = 0

        return x_batch, y_batch
