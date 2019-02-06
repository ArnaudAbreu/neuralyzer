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
import sys

# Dependency imports
from absl import flags
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.contrib.learn.python.learn.datasets import mnist

tfd = tfp.distributions

IMAGE_SHAPE = [28, 28, 1]

flags.DEFINE_float("learning_rate",
                   default=0.001,
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
FLAGS(sys.argv)


def build_input_pipeline(mnist_data, batch_size, heldout_size):

    """
    Build an Iterator switching between train and heldout data.
    """

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


mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

images, labels, handle, training_iterator, heldout_iterator = build_input_pipeline(mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)
