# coding: utf8

from matplotlib import pyplot as plt
from neuralyzer.data import patch
from openslide import OpenSlide

slide = OpenSlide('/Volumes/maxtor/charlotte/HyperPfolliculaires/CF_Margot-N1-200_16T000004-6.mrxs')

for im in patch.patches_in_slide(slide, 5, 125, 125, 125):
    plt.imshow(im)
    plt.show()
