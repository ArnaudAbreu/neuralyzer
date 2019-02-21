# coding: utf8

from tqdm import tqdm
import numpy
import argparse
import os
import pickle
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--predictiondir",
                    help="absolute path to a labpathfile produced by train-test split procedure")

parser.add_argument("--diagdir",
                    help="output folder for trained network")

parser.add_argument("--predtype",
                    help="one of the following: 'mean', 'med', 'maj' for slide diagnosis prediction")

args = parser.parse_args()

if not os.path.exists(args.diagdir):
    os.makedirs(args.diagdir)

diagfile = os.path.join(args.diagdir, 'diagnosis.diag')


# structure of a prediction file (slide):

# list of dico: [patch 1, patch 2, patch 3, ..., patch n]

# a patch is a dico: {'absizex': float,
#                     'absizey': float,
#                     'absx': float,
#                     'absy': float,
#                     'groundtruth': str ('LF' or 'HF'),
#                     'patchlevel': int,
#                     'prediction': list of arrays}

# the 'prediction' list of arrays is: [arrayofshape(0, 2), ...]

def get_patch_pred(patch, type='mean'):

    """
    """
    preds = numpy.asarray(patch['prediction'])

    if type == 'mean':
        return numpy.mean(preds, axis=0)

    if type == 'med':
        return numpy.median(preds, axis=0)


def get_patch_var(patch):

    """
    """
    preds = numpy.asarray(patch['prediction'])

    return numpy.var(preds[:, 0])


def get_patch_certainties(patch, varmax):

    """
    """

    return 1. - (get_patch_var(patch) / varmax)


def get_patch_certainty(patch, varmax):

    """
    """

    return numpy.mean(get_patch_certainties(patch, varmax))


def get_global_maxvar(predictionfiles):

    """
    """

    maxvar = 0.

    for prediction_file in predictionfiles:

        with open(prediction_file, 'rb') as f:
            patches = pickle.load(f)

        for patch in patches:

            maxvar = max([maxvar, get_patch_var(patch)])

    return maxvar


def get_sample_drawn(predictionfiles):

    """
    """

    with open(predictionfiles[0], 'rb') as f:
        patches = pickle.load(f)

    patch = patches[0]['prediction']

    return len(patch)


def get_slide_prediction(slide, varmax, thresh=0.5, type='mean'):

    """
    """

    patchpreds = []
    patchcert = []

    for patch in slide:

        patchpreds.append(get_patch_pred(patch))
        patchcert.append(get_patch_certainty(patch, varmax))

    patchpreds = numpy.asarray(patchpreds)
    patchcert = numpy.asarray(patchcert)

    if type == 'mean':
        return numpy.mean(patchpreds[patchcert > thresh], axis=0), numpy.mean(patchcert[patchcert > thresh])

    if type == 'med':
        return numpy.median(patchpreds[patchcert > thresh], axis=0), numpy.mean(patchcert[patchcert > thresh])

    if type == 'maj':
        cl = numpy.argmax(patchpreds, axis=1)
        hf = (cl == 0).sum()
        hfpercent = float(hf) / len(cl)
        return numpy.asarray([hfpercent, (1. - hfpercent)]), numpy.mean(patchcert[patchcert > thresh])


predfiles = [fname for fname in os.listdir(args.predictiondir) if fname[0] != '.' and '.p' in fname]

V = get_global_maxvar([os.path.join(args.predictiondir, p) for p in predfiles])

n_samples = get_sample_drawn([os.path.join(args.predictiondir, p) for p in predfiles])

print('\nprediction and certainty are computed on ', n_samples, ' samples averaged.\n')

globalpreds = []

for fname in tqdm(predfiles):

    predfile = os.path.join(args.predictiondir, fname)

    d = dict()

    with open(predfile, 'rb') as f:
        slide = pickle.load(f)

    pred, cert = get_slide_prediction(slide, V, type=args.predtype)

    d['name'] = os.path.splitext(fname)[0]
    d['prediction'] = pred
    d['patch_certainty'] = cert
    d['diag_certainty'] = abs(0.5 - max(pred)) / 0.5

    if slide[0]['groundtruth'] == 'HF':
        gt = numpy.array([1, 0], int)
    else:
        gt = numpy.array([0, 1], int)

    d['groundtruth'] = gt

    if numpy.argmax(pred) == numpy.argmax(gt):
        d['diagnosis'] = 'correct'

    else:
        d['diagnosis'] = 'incorrect'

    globalpreds.append(d)


baddiag = [elem for elem in globalpreds if elem['diagnosis'] == 'incorrect']
gooddiag = [elem for elem in globalpreds if elem['diagnosis'] == 'correct']

print('\ndiagnosis errors: ', len(baddiag), '\n')

for bd in baddiag:

    print('name: ', bd['name'], '\n',
          'prediction: ', bd['prediction'], '\n',
          'groundtruth: ', bd['groundtruth'], '\n',
          'patch_certainty: ', bd['patch_certainty'], '\n',
          'diag_certainty: ', bd['diag_certainty'], '\n',
          '#' * 20)
