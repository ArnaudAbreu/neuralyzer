import numpy
import pickle
import os
from skimage.io import imsave
from tqdm import trange, tqdm


def predict_slides(my_model, slidedir, outputdir, maxfiles=None, prediction_level=2, h_input=125, w_input=125, rescale=(1. / 255.), offsets=[(0, 0)], n_classes=3):

    """
    A function to predict WSI tiles with image classifier.
    (A smarter one)
    """

    from openslide import OpenSlide

    slidepaths = [os.path.join(slidedir, f) for f in os.listdir(slidedir) if f[0] != '.' and '.mrxs' in f]

    if maxfiles is not None:
        slidepaths = slidepaths[0:maxfiles]

    k = 0

    # for each slide
    for path in slidepaths:

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(slidepaths)) * 100, '%')
        print('#' * 20)

        # constant values
        level = prediction_level

        imagesize = (h_input, w_input)

        rescale_value = rescale

        slide = OpenSlide(path)

        slidex, slidey = slide.level_dimensions[level]
        tilex, tiley = imagesize

        nx = int(slidex / tilex)
        ny = int(slidey / tiley)

        stepx = tilex * (2 ** level)
        stepy = tiley * (2 ** level)

        # first, avoid empty patches
        # inform user:
        print('finding empty patches...')
        print('------------------------')
        useful_patches = dict()
        for x in trange(nx):
            useful_y = []
            for y in range(ny):
                # reasoning on red channel
                im = numpy.array(slide.read_region(location=(x * stepx, y * stepy), level=level, size=imagesize))[:, :, 0]
                if (im > 0).any():
                    useful_y.append(y)
            # if any y was good, I will process the x
            if useful_y:
                useful_patches[x] = useful_y

        # for each offset
        for offset in offsets:

            # inform user
            print('patch offset: ', offset)
            print('----------------------')

            result = numpy.zeros((ny, nx, n_classes))

            offsetx = offset[0]
            offsety = offset[1]

            try:

                # ---------------------
                # predicting iterations
                # ---------------------
                # predict column by column
                for x, ys in tqdm(useful_patches.items()):
                    # print("column ", (x + 1), "/", nx)
                    X = [numpy.array(slide.read_region(location=(offsetx + (x * stepx), offsety + (y * stepy)), level=level, size=imagesize))[:, :, 0:3].astype(float) * rescale_value for y in ys]
                    X = numpy.asarray(X)
                    x_pred = my_model.predict(X)
                    result[ys, x, :] = x_pred[0]

                result *= 255
                result = numpy.around(result)
                result[result > 255] = 255
                imsave(os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction_" + str(offsetx) + "_" + str(offsety) + ".tiff"), result)

            except KeyboardInterrupt:
                print("\nuser stopped iterations!")
                result *= 255
                result = numpy.around(result)
                result[result > 255] = 255
                imsave(os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction_" + str(offsetx) + "_" + str(offsety) + ".tiff"), result)
