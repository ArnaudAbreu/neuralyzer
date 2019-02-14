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


def get_stat_patch_pred(list):

    patch = numpy.asarray(list)

    return numpy.mean(patch, axis=0), numpy.var(patch, axis=0)


def get_average_slide_pred(prediction_file):

    with open(prediction_file, 'rb') as f:
        data = pickle.load(f)

    preds = [get_stat_patch_pred(elem['prediction']) for elem in data]
    preds_m = numpy.asarray([p[0] for p in preds])
    preds_v = numpy.asarray([p[1] for p in preds])

    if data[0]['groundtruth'] == 'HF':
        gt = numpy.array([1, 0], int)
    else:
        gt = numpy.array([0, 1], int)

    return preds_m.mean(axis=0), preds_v.mean(axis=0), gt


def get_median_slide_pred(prediction_file):

    with open(prediction_file, 'rb') as f:
        data = pickle.load(f)

    preds = [get_stat_patch_pred(elem['prediction']) for elem in data]
    preds_m = numpy.asarray([p[0] for p in preds])
    preds_v = numpy.asarray([p[1] for p in preds])

    if data[0]['groundtruth'] == 'HF':
        gt = numpy.array([1, 0], int)
    else:
        gt = numpy.array([0, 1], int)

    return numpy.median(preds_m, axis=0), numpy.median(preds_v, axis=0), gt


def get_majority_slide_pred(prediction_file):

    with open(prediction_file, 'rb') as f:
        data = pickle.load(f)

    preds = numpy.asarray([elem['prediction'] for elem in data])
    labpred = numpy.argmax(preds, axis=1)
    n_tiles = preds.shape[0]
    lfpercent = float(labpred.sum()) / float(n_tiles)
    hfpercent = 1. - lfpercent
    if data[0]['groundtruth'] == 'HF':
        gt = numpy.array([1, 0], int)
    else:
        gt = numpy.array([0, 1], int)
    return numpy.asarray([hfpercent, lfpercent]), gt


predictions = []
groundtruths = []
names = []
vars = []

for fname in tqdm(os.listdir(args.predictiondir)):

    if fname[0] != '.' and '.p' in fname:

        predfile = os.path.join(args.predictiondir, fname)
        if args.predtype == 'mean':
            pred, var, gt = get_average_slide_pred(predfile)
        elif args.predtype == 'med':
            pred, var, gt = get_median_slide_pred(predfile)
        else:
            pred, var, gt = get_majority_slide_pred(predfile)
        predictions.append(pred[0])
        vars.append(var[0])
        groundtruths.append(gt)
        names.append(fname)

predictions = numpy.asarray(predictions)
groundtruths = numpy.asarray(groundtruths)
vars = numpy.asarray(vars)
names = numpy.asarray(names)
abspreds = numpy.argmax(predictions, axis=1)
abstruth = numpy.argmax(groundtruths, axis=1)

print('diagnosis error: ', (abstruth != abspreds).sum(), '/', abstruth.shape[0])

# print(predictions[abstruth != abspreds])
# print(names[abstruth != abspreds])

for k in numpy.where(abstruth != abspreds)[0]:
    print('name: ', names[k], '\n',
          'prediction: ', predictions[k], '\n',
          'groundtruth: ', groundtruths[k], '\n',
          'variance: ', vars[k], '\n',
          '#' * 20)

# Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(2):
#     fpr[i], tpr[i], _ = roc_curve(groundtruths[:, i], predictions[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# plt.plot(fpr[0], tpr[0], color='darkorange', label='ROC curve HF (area = %0.2f)' % roc_auc[0])
# plt.plot(fpr[1], tpr[1], color='navy', label='ROC curve LF (area = %0.2f)' % roc_auc[1])
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic of CNN classifier')
# plt.legend(loc="lower right")
# plt.savefig(os.path.join(args.diagdir, 'roc_curves.svg'))
