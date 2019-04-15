import numpy
import pickle
import os
from skimage.io import imsave
from tqdm import trange, tqdm
from .patch import patches_in_slide
from pysliderois.tissue import PatchTree


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


def predict_slides_from_dir(my_model, slidedir, outputdir, patch_level, interval, x_size, y_size, maxfiles=None):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It put every results in an outputdir.
    This function takes advantage of the slight code written in the module patch.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - slidedir: str, absolute path to slide directory.
        - outputdir: str, absolute path to output directory.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - maxfiles: int, file number limit, if None, no limit, default is None.

    Returns:
        - nothing, just store results of each slide in a file located in outputdir.
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

        slide = OpenSlide(path)
        outputdata = []
        images = []

        for image, patch_level, absx, absy, absizex, absizey in patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=True):

            outputdata.append({'patchlevel': patch_level,
                               'absx': absx,
                               'absy': absy,
                               'absizex': absizex,
                               'absizey': absizey})
            images.append(image.astype(float) / 255.)

        preds = my_model.predict(images)

        for n in range(len(outputdata)):

            outputdata[n]['prediction'] = preds[0][n]

        outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction.p")

        with open(outfile, 'wb') as f:
            pickle.dump(outputdata, f)


def predict_slides_from_labpathlist(my_model, labpathlist, outputdir, patch_level, interval, x_size, y_size, maxfiles=None):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It put every results in an outputdir.
    This function takes advantage of the slight code written in the module patch.
    It takes a list of filepaths, not a slide dir.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - labpathlist: list of tuples (str, str), absolute path to slides, with GT label.
        - outputdir: str, absolute path to output directory.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - maxfiles: int, file number limit, if None, no limit, default is None.

    Returns:
        - nothing, just store results of each slide in a file located in outputdir.
    """

    from openslide import OpenSlide

    k = 0

    # for each slide
    for labpath in labpathlist:

        path, lab = labpath

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(labpathlist)) * 100, '%')
        print('#' * 20)

        slide = OpenSlide(path)
        outputdata = []
        images = []

        for image, patch_level, absx, absy, absizex, absizey in patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=True):

            outputdata.append({'patchlevel': patch_level,
                               'absx': absx,
                               'absy': absy,
                               'absizex': absizex,
                               'absizey': absizey,
                               'groundtruth': lab})
            images.append(image.astype(float) / 255.)

        preds = my_model.predict(numpy.asarray(images))

        for n in range(len(outputdata)):

            outputdata[n]['prediction'] = preds[0][n]

        outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction.p")

        with open(outfile, 'wb') as f:
            pickle.dump(outputdata, f)


def sample_predict_slides_from_labpathlist(my_model, labpathlist, outputdir, patch_level, interval, x_size, y_size, batchsize=None, maxfiles=None):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It put every results in an outputdir.
    This function takes advantage of the slight code written in the module patch.
    It takes a list of filepaths, not a slide dir.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - labpathlist: list of tuples (str, str), absolute path to slides, with GT label.
        - outputdir: str, absolute path to output directory.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - maxfiles: int, file number limit, if None, no limit, default is None.

    Returns:
        - nothing, just store results of each slide in a file located in outputdir.
    """

    from openslide import OpenSlide

    k = 0

    # for each slide
    for labpath in labpathlist:

        path, lab = labpath

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(labpathlist)) * 100, '%')
        print('#' * 20)

        slide = OpenSlide(path)
        outputdata = []
        images = []

        for image, patch_level, absx, absy, absizex, absizey in patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=True):

            outputdata.append({'patchlevel': patch_level,
                               'absx': absx,
                               'absy': absy,
                               'absizex': absizex,
                               'absizey': absizey,
                               'groundtruth': lab})
            images.append(image.astype(float) / 255.)

        # if batchsize is None:
        #     preds = my_model.sample_predict(numpy.asarray(images))
        # else:
        #     preds = []
        #     batches = int(float(len(images)) / float(batchsize))
        #     for k in range(batches):
        #         inputs = numpy.asarray(images[k * batchsize:k * batchsize + batchsize])
        #         preds += my_model.sample_predict(inputs)

        for n in range(len(images)):

            image = images[n]

            outputdata[n]['prediction'] = my_model.sample_predict([image])

        # final_outputdata = [d for d in outputdata if 'prediction' in d.keys()]

        outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction.p")

        with open(outfile, 'wb') as f:
            pickle.dump(outputdata, f)


def dropout_predict_slides_from_labpathlist(my_model, labpathlist, outputdir, patch_level, interval, x_size, y_size, maxfiles=None):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It put every results in an outputdir.
    This function takes advantage of the slight code written in the module patch.
    It takes a list of filepaths, not a slide dir.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - labpathlist: list of tuples (str, str), absolute path to slides, with GT label.
        - outputdir: str, absolute path to output directory.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - maxfiles: int, file number limit, if None, no limit, default is None.

    Returns:
        - nothing, just store results of each slide in a file located in outputdir.
    """

    from openslide import OpenSlide

    k = 0

    # for each slide
    for labpath in labpathlist:

        path, lab = labpath

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(labpathlist)) * 100, '%')
        print('#' * 20)

        slide = OpenSlide(path)
        outputdata = []
        images = []

        for image, patch_level, absx, absy, absizex, absizey in patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=True):

            outputdata.append({'patchlevel': patch_level,
                               'absx': absx,
                               'absy': absy,
                               'absizex': absizex,
                               'absizey': absizey,
                               'groundtruth': lab})
            images.append(image.astype(float) / 255.)

        preds = my_model.predict(numpy.asarray(images))

        for n in range(len(outputdata)):

            outputdata[n]['prediction'] = [p[0][n] for p in preds]

        outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction.p")

        with open(outfile, 'wb') as f:
            pickle.dump(outputdata, f)


def dropout_predict_slides_from_dir(my_model, dirname, outputdir, patch_level, interval, x_size, y_size, maxfiles=None):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It puts every results in an outputdir.
    This function takes advantage of the slight code written in the module patch.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - dirname: str, absolute path to a directory containing mrxs slides.
        - outputdir: str, absolute path to output directory.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - maxfiles: int, file number limit, if None, no limit, default is None.

    Returns:
        - nothing, just store results of each slide in a file located in outputdir.
    """

    from openslide import OpenSlide

    k = 0

    namelist = [name for name in os.listdir(dirname) if '.mrxs' in name and name[0] != '.']
    pathlist = [os.path.join(dirname, name) for name in namelist]

    # for each slide
    for path in pathlist:

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(pathlist)) * 100, '%')
        print('#' * 20)

        slide = OpenSlide(path)
        outputdata = []
        images = []

        for image, patch_level, absx, absy, absizex, absizey in patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=True):

            outputdata.append({'patchlevel': patch_level,
                               'absx': absx,
                               'absy': absy,
                               'absizex': absizex,
                               'absizey': absizey})
            images.append(image.astype(float) / 255.)

        preds = my_model.predict(numpy.asarray(images))

        for n in range(len(outputdata)):

            outputdata[n]['prediction'] = [p[0][n] for p in preds]

        outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_prediction.p")

        with open(outfile, 'wb') as f:
            pickle.dump(outputdata, f)


def dropout_predict_patch_tree(my_model, patchtree, level):

    """
    Function to predict tiles from a patchtree. The patchtree is a pysliderois object.
    It has a hierarchical structure and generators to yield batch of patches.
    One should store prediction results inside the patchtree structure.

    Arguments:
        - my_model: classifier, object with a predict function that works on
        RGB-images.
        - patchtree: PatchTree object from pysliderois.
        - level: level to extract patches.
    """

    for patchbatch in patchtree.images_at_level(level):

        relpos, abspos, images = patchbatch

        preds = my_model.predict(images)

        for n in range(len(abspos)):

            locpreds = numpy.array([p[0][n] for p in preds])

            patchtree.predictions[abspos[n]] = numpy.mean(locpreds, axis=0)

            patchtree.variances[abspos[n]] = numpy.var(locpreds, axis=0)[0]


def dropout_predict_slides_from_dir_with_tree(my_models, dirname, outputdir, levelmax, levelmin):

    """
    Same function as above, take a slide dir and predict every tile of every
    slide in that dir. It puts every results in an outputdir.
    This function takes advantage of a complex code written in the package
    pysliderois for hierarchical patch computation.

    Arguments:
        - my_models: objects with a load function that works on to load trained
        classifiers that work on RGB-images.
        - dirname: str, absolute path to a directory containing mrxs slides.
        - outputdir: str, absolute path to output directory.
    """

    from openslide import OpenSlide

    k = 0

    namelist = [name for name in os.listdir(dirname) if '.mrxs' in name and name[0] != '.']
    pathlist = [os.path.join(dirname, name) for name in namelist]

    # for each slide
    for path in pathlist:

        k += 1

        # inform user
        print('#' * 20)
        print('Processing file: ', path)
        print('progression : ', (k / len(pathlist)) * 100, '%')
        print('#' * 20)

        slide = OpenSlide(path)

        # create a patch tree (pysliderois object)
        patchtree = PatchTree(slide, my_models.patchsize, levelmax, levelmin)

        # for each level in the patch tree
        for n in range(levelmin, levelmax + 1):

            # instanciate a model
            my_model = my_models.load_level(n)

            # predict with the model
            dropout_predict_patch_tree(my_model, patchtree, n)

            # store the patchtree
            outfile = os.path.join(outputdir, os.path.basename(path).split(".")[0] + "_patchtree.p")

            with open(outfile, 'wb') as f:
                pickle.dump(patchtree, f)

            my_model.close()
