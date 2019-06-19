# coding: utf8
import numpy
from skimage.util import regular_seeds
from skimage.io import imsave
from skimage.exposure import is_low_contrast
import os
from tqdm import tqdm


def non_black(image, tol):

    """
    Given a color image, masks all black pixels.

    Arguments:
        - image: numpy array uint, (x, y, 3) rgb image.
        - tol: int, tolerance value for 0 intensity pixels.

    Returns:
        - mask: numpy array bool, (x, y) binary mask.
    """

    mask = numpy.ones((image.shape[0], image.shape[1]))

    for channel in range(image.shape[-1]):

        mask = numpy.logical_and(mask, image[:, :, channel] > (0 + tol))

    return mask


def non_white(image, tol):

    """
    Given a color image, masks all white pixels.

    Arguments:
        - image: numpy array uint, (x, y, 3) rgb image.
        - tol: int, tolerance value for 255 intensity pixels.

    Returns:
        - mask: numpy array bool, (x, y) binary mask.
    """

    mask = numpy.ones((image.shape[0], image.shape[1]))

    for channel in range(image.shape[-1]):

        mask = numpy.logical_and(mask, image[:, :, channel] < (255 - tol))

    return mask


def tissue_mask(image, tol_black=0, tol_white=75):

    """
    Given a color image, masks all white and black pixels to focus on tissue.

    Arguments:
        - image: numpy array uint, (x, y, 3) rgb image.
        - tol_black: int, tolerance value for 0 intensity pixels.
        - tol_white: int, tolerance value for 255 intensity pixels.

    Returns:
        - mask: numpy array bool, (x, y) binary mask, tissue in white.
    """

    mask = numpy.logical_and(non_black(image, tol_black), non_white(image, tol_white))

    return mask


def get_size_at_level(size, inlevel, outlevel):

    """
    Given a size at inlevel, returns size at outlevel.

    Arguments:
        - size: int, size value.
        - inlevel: int, pyramid level index where size = size.
        - outlevel: int, pyramid level index where size != size.

    Returns:
        - new_size: int, value of size at level outlevel.
    """

    absize = size * (2 ** inlevel)

    new_size = absize / (2 ** outlevel)

    return new_size


def get_reasonable_level(slide):

    """
    Given a slide, find the best level to process in RAM painlessly.

    Arguments:
        - slide: OpenSlide object.

    Returns:
        - k: int, best resolution with di and dj < 10000 pixels.
    """

    level_dimensions = list(slide.level_dimensions)
    k = 0
    # level dimensions are listed in decreasing order.

    for di, dj in level_dimensions:

        if di < 10000 and dj < 10000:
            break
        else:
            k += 1

    return k


def regular_grid_regular_seeds(imshape, interval):

    """
    Given the shape of an image and an interval, returns seed array. One value
    per seed. Seed array: one pixel per seed.

    Arguments:
        - imshape: tuple, shape of output seed map.
        - interval: int, space sampling distance.

    Returns:
        - seed_array: numpy array, one int value for each seed, background = 0.
    """

    n_seeds = 1
    for s in imshape:
        n_seeds *= int(s / interval)

    return regular_seeds(imshape, n_seeds)


def patches_in_slide(slide, patch_level, interval, x_size, y_size, detailed=False):

    """
    Given a slide, a level to extract patches, sampling intervals, sample sizes,
    computes the ROIs (its an iterator).

    Arguments:
        - slide: OpenSlide object.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.
        - detailed: bool, whether to return position and size of patch in wsi.

    Yields:
        - image: RGB numpy array.
    """

    sampling_level = get_reasonable_level(slide)

    print('sampling level: ', sampling_level)

    interval_sampling_level = int(get_size_at_level(interval, patch_level, sampling_level))

    print('interval sampling level: ', interval_sampling_level)

    image_sampling_level = numpy.array(slide.read_region((0, 0), sampling_level, slide.level_dimensions[sampling_level]))[:, :, 0:3]

    unmasked_seeds = regular_grid_regular_seeds((slide.level_dimensions[sampling_level][1], slide.level_dimensions[sampling_level][0]), interval_sampling_level)

    tissue = tissue_mask(image_sampling_level)

    masked_seeds = numpy.logical_and(unmasked_seeds, tissue)

    i, j = numpy.where(masked_seeds)

    print('number of patches: ', len(i))

    for y, x in zip(i, j):

        absx = get_size_at_level(x, sampling_level, 0)
        absy = get_size_at_level(y, sampling_level, 0)

        absizex = get_size_at_level(x_size, patch_level, 0)
        absizey = get_size_at_level(y_size, patch_level, 0)

        startx = max(0, int(absx - 0.5 * absizex))
        starty = max(0, int(absy - 0.5 * absizey))

        image = numpy.array(slide.read_region((startx, starty), patch_level, (x_size, y_size)))

        # still have to check whether is_low_contrast(image) because white and black elimination
        # was performed at lower resolution, forgotten areas might remain.

        if not is_low_contrast(image[:, :, 0:3]):

            if detailed:
                yield image[:, :, 0:3], patch_level, absx, absy, absizex, absizey
            else:
                yield image[:, :, 0:3]


def patchify(slide, patch_level, interval, x_size, y_size, prefix):

    """
    Given a slide, a level to extract patches, sampling intervals, sample sizes,
    computes the ROIs (its an iterator).

    Arguments:
        - slide: OpenSlide object.
        - patch_level: int, level to extract patches, 0 = highest resolution.
        - interval: int, number of pixels between two samples given at patch level.
        - x_size: int, number of pixels of sample on x axis, given at patch level.
        - y_size: int, number of pixels of sample on y axis given at patch level.

    Yields:
        - image: RGB numpy array.
    """

    sampling_level = get_reasonable_level(slide)

    # print('sampling level: ', sampling_level)

    interval_sampling_level = int(get_size_at_level(interval, patch_level, sampling_level))

    # print('interval sampling level: ', interval_sampling_level)

    image_sampling_level = numpy.array(slide.read_region((0, 0), sampling_level, slide.level_dimensions[sampling_level]))[:, :, 0:3]

    unmasked_seeds = regular_grid_regular_seeds((slide.level_dimensions[sampling_level][1], slide.level_dimensions[sampling_level][0]), interval_sampling_level)

    tissue = tissue_mask(image_sampling_level)

    masked_seeds = numpy.logical_and(unmasked_seeds, tissue)

    # plt.imshow(masked_seeds)
    # plt.show()

    i, j = numpy.where(masked_seeds)

    # print('number of patches: ', len(i))

    for y, x in zip(i, j):

        absx = get_size_at_level(x, sampling_level, 0)
        absy = get_size_at_level(y, sampling_level, 0)

        absizex = get_size_at_level(x_size, patch_level, 0)
        absizey = get_size_at_level(y_size, patch_level, 0)

        startx = max(0, int(absx - 0.5 * absizex))
        starty = max(0, int(absy - 0.5 * absizey))

        image = numpy.array(slide.read_region((startx, starty), patch_level, (x_size, y_size)))

        if not is_low_contrast(image[:, :, 0:3]):

            outpath = prefix + "_" + str(x) + "_" + str(y) + ".png"

            # outpath = os.path.join(prefix, name)

            imsave(outpath, image[:, :, 0:3])
