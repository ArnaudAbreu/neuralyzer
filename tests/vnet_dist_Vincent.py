# coding: utf8
import tensorflow as tf
import math
from neuralyzer.archi import Vnet_4levels
import argparse
import numpy
import shutil
import json
import scipy
import scipy.ndimage as ndi
import time
from sklearn.feature_extraction import image
import keras
from keras.layers import Input, Concatenate, concatenate, Conv2D, Dropout, BatchNormalization, ELU, Activation, PReLU, SpatialDropout2D, LeakyReLU, Multiply, Subtract, Conv2DTranspose, Add
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os


# this file is a test for all functionalities of the deep ensemble
parser = argparse.ArgumentParser()

parser.add_argument("--batchsize", type=int, default=28,
                    help="int, size of batches")

parser.add_argument("--epochs", type=int, default=300,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--outfolder",
                    help="output folder for trained network")

parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate for training")

parser.add_argument("--dropout", type=float, default=0.5,
                    help="learning rate for training")

parser.add_argument("--indir",
                    help="data directory")

parser.add_argument("--size", type=int, default=64,
                    help="patch size")

parser.add_argument("--ncat", type=int, default=12,
                    help="number of categories")

parser.add_argument("-weighted", default='no',
                    help='is weighting applied on classes in loss')

args = parser.parse_args()
batchsize = args.batchsize
epochs = args.epochs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

if 'n' in args.weighted.lower():
    WEIGHTED = False
else:
    WEIGHTED = True


DIR_FOR_TRAIN = args.outfolder

# DANGER: THIS CLEAR ALL THE SAVED WORK
if os.path.exists(DIR_FOR_TRAIN):
    shutil.rmtree(DIR_FOR_TRAIN)

# Output folders creation
os.makedirs(DIR_FOR_TRAIN)
os.makedirs(DIR_FOR_TRAIN + "/weights")
os.makedirs(DIR_FOR_TRAIN + "/fig")

DIR_FOR_DATA = args.indir

PATCH_DIM = args.size
memo_train = {}
memo_train["DIR_FOR_DATA"] = DIR_FOR_DATA

memo_train["DIR_FOR_TRAIN"] = DIR_FOR_TRAIN
memo_train["PATCH_DIM"] = PATCH_DIM
memo_train["nb_cat"] = args.ncat

memo_train["dropout"] = args.dropout


def loadOneKindOfData(trainOrTest):
    X = numpy.load(os.path.join(DIR_FOR_DATA, trainOrTest + "/X/images.npy"))
    Y = numpy.load(os.path.join(DIR_FOR_DATA, trainOrTest + "/Y/labels.npy"))

    return X, Y


def getPatchesForOneEpoch(Xs, Ys, nbPatches=16, seed=None):

    if seed is None:
        usedSeed = numpy.random.randint(2**32 - 1)
    else:
        usedSeed = seed

    print("random-seed for shuffle and patch-sampling:" + str(usedSeed))
    numpy.random.seed(seed)

    res_X = []
    res_Y = []

    # we take nbPatches in each image
    for X, Y in zip(Xs, Ys):
        # the two random_state must have the same seed
        xs = image.extract_patches_2d(X, (PATCH_DIM, PATCH_DIM), max_patches=nbPatches, random_state=usedSeed)
        ys = image.extract_patches_2d(Y, (PATCH_DIM, PATCH_DIM), max_patches=nbPatches, random_state=usedSeed)

        for x, y in zip(xs, ys):
            # if no pixel are outside
            if numpy.sum(y < 0.5) == 0:
                res_X.append(x)
                res_Y.append(y[:, :] - 1)

    res_X = numpy.array(res_X)
    res_Y = numpy.array(res_Y)
    perm = numpy.random.permutation(len(res_X))
    res_X_perm = res_X[perm]
    res_Y_perm = res_Y[perm]

    return res_X_perm, res_Y_perm


def head_classif_multilabel(Y, nbClasses):
    return Conv2D(nbClasses, 1, activation='sigmoid')(Y)


def weighted_dice_coef_multiD_tf(y_true, y_pred, weight, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred, axis=[0, 1]) + smooth)
    wdice = dice * weight
    # sum over channels
    return tf.reduce_mean(wdice)


def dice_coef_multiD_tf(y_true, y_pred, smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[0, 1])
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true, axis=[0, 1]) + tf.reduce_sum(y_pred, axis=[0, 1]) + smooth)
    # sum over channels
    return tf.reduce_mean(dice)


def weighted_dice_loss_multi_D(y_true, y_pred, weight):
    smooth = 1e-5
    return 1. - weighted_dice_coef_multiD_tf(y_true, y_pred, weight, smooth)


def dice_loss_multi_D(y_true, y_pred):
    smooth = 1e-5
    return 1. - dice_coef_multiD_tf(y_true, y_pred, smooth)


weight = tf.range(start=memo_train['nb_cat'], limit=0, delta=-1, dtype=tf.float32) / float(memo_train['nb_cat'])


def make_model():

    inputs = keras.Input(batch_shape=(None, PATCH_DIM, PATCH_DIM, 3))

    Y = Vnet_4levels(memo_train["dropout"])(inputs)

    """premier essais: pas terrible!"""
    # output = head_regression(Y)
    # loss = square_loss

    # I think that we can reduce the 'nb_cat' because at the end, the model to not recognise all of them
    output = head_classif_multilabel(Y, memo_train["nb_cat"])

    if WEIGHTED:
        def m_loss(y_true, y_pred):
            return weighted_dice_loss_multi_D(y_true, y_pred, weight)

        loss = m_loss

    else:
        loss = dice_loss_multi_D

    memo_train["loss"] = "dice_loss_multi_D"

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss=loss)
    return model, loss


def to_categorical_pacth_overlaping(y, nb_cat):

    y_cat = numpy.zeros([y.shape[0], args.size, args.size, nb_cat], dtype=numpy.int32)
    y_cat[:, :, :, 0] = (y == 0).astype(numpy.int32)

    for i in range(1, nb_cat):
        y_cat[:, :, :, i] = (y >= i).astype(numpy.int32)

    return y_cat


def reconstitution(y):

    return numpy.sum(y[:, :, :, 1:], axis=3)


X_test, Y_test = loadOneKindOfData("test")
X_train, Y_train = loadOneKindOfData("train")

# this test set will serve all the validation steps
X_test_patch, Y_test_patch = getPatchesForOneEpoch(X_test, Y_test, nbPatches=40, seed=213)
Y_test_patch = Y_test_patch[:, :, :]

print("shape of Y_test_patch:", Y_test_patch.shape)

Y_test_patch_cat = to_categorical_pacth_overlaping(Y_test_patch, memo_train["nb_cat"])

print("Y_test_patch_cat:", Y_test_patch_cat.shape)


loss = []
val_loss = []
nb_epochs = 0
duration = 0
total_epochs = 300
model, _ = make_model()

starting_time = time.time()

try:

    for e in range(total_epochs):

        print("epoch nb: " + str(nb_epochs))

        sess = tf.Session()
        print('weight for the loss: ', sess.run(weight))

        X_train_patch, Y_train_patch = getPatchesForOneEpoch(X_train, Y_train)
        Y_train_patch_cat = to_categorical_pacth_overlaping(Y_train_patch, memo_train["nb_cat"])

        history = model.fit(X_train_patch,
                            Y_train_patch_cat,
                            batch_size=args.batchsize,  # 40
                            # initial_epoch=nb_epochs,
                            # epochs=nb_epochs+1,
                            validation_data=(X_test_patch, Y_test_patch_cat)
                            )

        loss.append(history.history["loss"][0])
        val_loss.append(history.history["val_loss"][0])
        nb_epochs += 1

except KeyboardInterrupt:
    print("interupted")

ending_time = time.time()
duration += ending_time - starting_time

memo_train["nb_epochs"] = nb_epochs
memo_train["training_duration"] = duration
memo_train["loss"] = loss
memo_train["val_loss"] = val_loss

fig, ax = plt.subplots()
ax.plot(range(memo_train["nb_epochs"]), (numpy.array(loss)), label="loss")
ax.plot(range(memo_train["nb_epochs"]), (numpy.array(val_loss)), label="val_loss")
ax.legend()
fig.savefig(DIR_FOR_TRAIN + "/fig/loss.png")

with open(DIR_FOR_TRAIN + '/memo_train.json', 'w') as fp:
    json.dump(memo_train, fp)

hat_Y_test_patch_cat = model.predict(X_test_patch)
hat_Y_test_patch_recons = reconstitution(hat_Y_test_patch_cat)
print(hat_Y_test_patch_recons.shape, hat_Y_test_patch_recons.max())

nb = 10
fig, axs = plt.subplots(nb, 3, figsize=(10, 20))
print(axs.shape)
numpy.random.seed(40)
perm = numpy.random.permutation(len(X_test_patch))

for i in range(nb):
    j = perm[i]
    x, y, hat_y = X_test_patch[j], Y_test_patch[j], hat_Y_test_patch_recons[j]
    axs[i, 0].imshow(x, vmin=0, vmax=1)
    axs[i, 1].imshow(y[:, :])
    axs[i, 2].imshow(hat_y[:, :])

fig.savefig(DIR_FOR_TRAIN + "/fig/comparison_seed40_epoch" + str(nb_epochs))


def saveModel(model):

    json_string = model.to_json()

    with open(DIR_FOR_TRAIN + '/weights/model.json', "w") as text_file:
        text_file.write(json_string)


saveModel(model)

model.save_weights(DIR_FOR_TRAIN + '/weights/epoch' + str(nb_epochs) + '.h5')
