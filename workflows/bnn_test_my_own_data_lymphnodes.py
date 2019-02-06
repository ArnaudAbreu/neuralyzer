# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify MNIST digits.

The architecture is LeNet-5 [1].

#### References

[1]: Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
     Gradient-based learning applied to document recognition.
     _Proceedings of the IEEE_, 1998.
     http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
from absl import flags
import matplotlib
matplotlib.use("Agg")
from matplotlib import figure  # pylint: disable=g-import-not-at-top
from matplotlib.backends import backend_agg
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib.learn.python.learn.datasets import mnist

from neuralyzer.archi import BayesianClassifier
from neuralyzer.data import FolderGenerator as foldgen

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--level", type=int, default=5,
                    help="int, pyramid level, sample resolution.")

parser.add_argument("--interval", type=int, default=125,
                    help="int, interval between two samples, at level=--level.")

parser.add_argument("--sizex", type=int, default=125,
                    help="int, size of sample on x axis, at level=--level.")

parser.add_argument("--sizey", type=int, default=125,
                    help="int, size of sample on y axis, at level=--level.")

parser.add_argument("--trainratio", type=float, default=0.66,
                    help="float, percentage of data to put in training set.")

parser.add_argument("--infolder", type=str, help="path to slide folder.")

parser.add_argument("--outfolder", type=str, help="path to outfolder.")

parser.add_argument("--batchsize", type=int, default=10,
                    help="int, size of batches")

parser.add_argument("--epochs", type=int, default=100,
                    help="number of epochs for training")

parser.add_argument("--device", default="0",
                    help="device to use for computation")

parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")

parser.add_argument("--opt", default="SGD",
                    help="optimizer")

args = parser.parse_args()

outpatchfolder = os.path.join(args.outfolder, 'Data')
train_folder = os.path.join(outpatchfolder, 'Train')
valid_folder = os.path.join(outpatchfolder, 'Test')
network_folder = os.path.join(args.outfolder, 'Model')
labpathfile = os.path.join(args.outfolder, 'labpathlist.p')
network_file = os.path.join(network_folder, 'model.ckpt')
eval_folder = os.path.join(args.outfolder, 'Evaluation')

# TODO(b/78137893): Integration tests currently fail with seaborn imports.
warnings.simplefilter(action="ignore")

try:
  import seaborn as sns  # pylint: disable=g-import-not-at-top
  HAS_SEABORN = True
except ImportError:
  HAS_SEABORN = False

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float("learning_rate",
                   default=0.0001,
                   help="Initial learning rate.")
flags.DEFINE_integer("max_steps",
                     default=6000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("batch_size",
                     default=128,
                     help="Batch size.")
flags.DEFINE_string("data_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "bayesian_neural_network/data"),
                    help="Directory where data is stored (if using real data).")
flags.DEFINE_string(
    "model_dir",
    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                         "bayesian_neural_network/"),
    help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps",
                     default=400,
                     help="Frequency at which save visualizations.")
flags.DEFINE_integer("num_monte_carlo",
                     default=50,
                     help="Network draws to compute predictive probabilities.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If true, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS


def build_input_pipeline(mnist_data, batch_size, heldout_size):
    """Build an Iterator switching between train and heldout data."""

    # Build an iterator over training batches.
    training_dataset = tf.data.Dataset.from_tensor_slices((mnist_data.train.images, np.int32(mnist_data.train.labels)))
    training_batches = training_dataset.shuffle(50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_iterator = training_batches.make_one_shot_iterator()

    # Build a iterator over the heldout set with batch_size=heldout_size,
    # i.e., return the entire heldout set as a constant.
    heldout_dataset = tf.data.Dataset.from_tensor_slices((mnist_data.validation.images, np.int32(mnist_data.validation.labels)))
    heldout_frozen = (heldout_dataset.take(heldout_size).repeat().batch(heldout_size))
    heldout_iterator = heldout_frozen.make_one_shot_iterator()

    # Combine these into a feedable iterator that can switch between training
    # and validation inputs.
    handle = tf.placeholder(tf.string, shape=[])
    feedable_iterator = tf.data.Iterator.from_string_handle(handle, training_batches.output_types, training_batches.output_shapes)
    images, labels = feedable_iterator.get_next()

    return images, labels, handle, training_iterator, heldout_iterator


def main(argv):

    del argv

    train_data = foldgen(train_folder, args.batchsize, size=(args.sizex, args.sizey), class_mode='sparse')
    valid_data = foldgen(valid_folder, args.batchsize, size=(args.sizex, args.sizey), class_mode='sparse')

    images = tf.placeholder(dtype=tf.float32, shape=(None, args.sizex, args.sizey, 3), name='input')
    labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')

    neural_net = BayesianClassifier(brickname='reference',
                                    filters=[32, 64, 128],
                                    kernels=[4, 5, 6],
                                    strides=[1, 1, 1],
                                    dropouts=[0.1, 0.2, 0.25],
                                    fc=[1024, 1024],
                                    fcdropouts=[0.5, 0.5],
                                    conv_activations=['relu', 'relu', 'relu'],
                                    fc_activations=['relu', 'relu'],
                                    end_activation=None,
                                    output_channels=2)

    logits = neural_net(images)
    labels_distribution = tfd.Categorical(logits=logits)

    # Compute the -ELBO as the loss, averaged over the batch size.
    neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
    kl = sum(neural_net.losses) / args.batchsize
    elbo_loss = neg_log_likelihood + kl

    # Build metrics for evaluation. Predictions are formed from a single forward
    # pass of the probabilistic layers. They are cheap but noisy predictions.
    predictions = tf.argmax(logits, axis=1)
    accuracy, accuracy_update_op = tf.metrics.accuracy(labels=labels, predictions=predictions)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(elbo_loss)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        # Run the training loop.
        # train_handle = sess.run(training_iterator.string_handle())
        # heldout_handle = sess.run(heldout_iterator.string_handle())
        step = 0
        for x, y in train_data:
            _ = sess.run([train_op, accuracy_update_op], feed_dict={images: x, labels: y})
            if step % 100 == 0:
                imvals, labvals, loss_value, accuracy_value = sess.run([images, labels, elbo_loss, accuracy], feed_dict={images: x, labels: y})
                print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(step, loss_value, accuracy_value))

            # print('images: ', imvals)
            # print('labels: ', labvals)
            step += 1


if __name__ == "__main__":
    tf.app.run()
