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

def build_input_pipeline(mnist_data, batch_size, heldout_size):
  """Build an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.train.images, np.int32(mnist_data.train.labels)))
  training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = training_batches.make_one_shot_iterator()

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (mnist_data.validation.images,
       np.int32(mnist_data.validation.labels)))
  heldout_frozen = (heldout_dataset.take(heldout_size).
                    repeat().batch(heldout_size))
  heldout_iterator = heldout_frozen.make_one_shot_iterator()

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.placeholder(tf.string, shape=[])
  feedable_iterator = tf.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def build_fake_data(num_examples=10):
  """Build fake MNIST-style data for unit testing."""

  class Dummy(object):
    pass

  num_examples = 10
  mnist_data = Dummy()
  mnist_data.train = Dummy()
  mnist_data.train.images = np.float32(np.random.randn(
      num_examples, *IMAGE_SHAPE))
  mnist_data.train.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.train.num_examples = num_examples
  mnist_data.validation = Dummy()
  mnist_data.validation.images = np.float32(np.random.randn(
      num_examples, *IMAGE_SHAPE))
  mnist_data.validation.labels = np.int32(np.random.permutation(
      np.arange(num_examples)))
  mnist_data.validation.num_examples = num_examples
  return mnist_data


def main(argv):
  del argv  # unused
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    mnist_data = build_fake_data()
  else:
    mnist_data = mnist.read_data_sets(FLAGS.data_dir, reshape=False)

  (images, labels, handle,
   training_iterator, heldout_iterator) = build_input_pipeline(
       mnist_data, FLAGS.batch_size, mnist_data.validation.num_examples)

  # Build a Bayesian LeNet5 network. We use the Flipout Monte Carlo estimator
  # for the convolution and fully-connected layers: this enables lower
  # variance stochastic gradients than naive reparameterization.

  neural_net = BayesianClassifier(brickname='reference',
                                  filters=[6, 16, 120],
                                  kernels=[5, 5, 5],
                                  strides=[1, 1, 1],
                                  dropouts=[0.1, 0.2, 0.25],
                                  fc=[84],
                                  fcdropouts=[0.5],
                                  conv_activations=['relu', 'relu', 'relu'],
                                  fc_activations=['relu', 'relu'],
                                  end_activation=None,
                                  output_channels=10)

  # with tf.name_scope("bayesian_neural_net", values=[images]):
  #   neural_net = tf.keras.Sequential([
  #       tfp.layers.Convolution2DFlipout(6,
  #                                       kernel_size=5,
  #                                       padding="SAME",
  #                                       activation=tf.nn.relu),
  #       tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
  #                                    strides=[2, 2],
  #                                    padding="SAME"),
  #       tfp.layers.Convolution2DFlipout(16,
  #                                       kernel_size=5,
  #                                       padding="SAME",
  #                                       activation=tf.nn.relu),
  #       tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
  #                                    strides=[2, 2],
  #                                    padding="SAME"),
  #       tfp.layers.Convolution2DFlipout(120,
  #                                       kernel_size=5,
  #                                       padding="SAME",
  #                                       activation=tf.nn.relu),
  #       tf.keras.layers.Flatten(),
  #       tfp.layers.DenseFlipout(84, activation=tf.nn.relu),
  #       tfp.layers.DenseFlipout(10)
  #       ])

  logits = neural_net(images)
  labels_distribution = tfd.Categorical(logits=logits)

  # Compute the -ELBO as the loss, averaged over the batch size.
  neg_log_likelihood = -tf.reduce_mean(labels_distribution.log_prob(labels))
  kl = sum(neural_net.losses) / mnist_data.train.num_examples
  elbo_loss = neg_log_likelihood + kl

  # Build metrics for evaluation. Predictions are formed from a single forward
  # pass of the probabilistic layers. They are cheap but noisy predictions.
  predictions = tf.argmax(logits, axis=1)
  accuracy, accuracy_update_op = tf.metrics.accuracy(
      labels=labels, predictions=predictions)

  # Extract weight posterior statistics for layers with weight distributions
  # for later visualization.
  # names = []
  # qmeans = []
  # qstds = []
  # for i, layer in enumerate(neural_net.layers):
  #   try:
  #     q = layer.kernel_posterior
  #   except AttributeError:
  #     continue
  #   names.append("Layer {}".format(i))
  #   qmeans.append(q.mean())
  #   qstds.append(q.stddev())

  with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(elbo_loss)

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  with tf.Session() as sess:
    sess.run(init_op)

    # Run the training loop.
    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    for step in range(FLAGS.max_steps):
      _ = sess.run([train_op, accuracy_update_op],
                   feed_dict={handle: train_handle})

      if step % 100 == 0:
        loss_value, accuracy_value = sess.run(
            [elbo_loss, accuracy], feed_dict={handle: train_handle})
        print("Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f}".format(
            step, loss_value, accuracy_value))

      # if (step+1) % FLAGS.viz_steps == 0:
      #   # Compute log prob of heldout set by averaging draws from the model:
      #   # p(heldout | train) = int_model p(heldout|model) p(model|train)
      #   #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
      #   # where model_i is a draw from the posterior p(model|train).
      #   probs = np.asarray([sess.run((labels_distribution.probs),
      #                                feed_dict={handle: heldout_handle})
      #                       for _ in range(FLAGS.num_monte_carlo)])
      #   mean_probs = np.mean(probs, axis=0)
      #
      #   image_vals, label_vals = sess.run((images, labels),
      #                                     feed_dict={handle: heldout_handle})
      #   heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
      #                                          label_vals.flatten()]))
      #   print(" ... Held-out nats: {:.3f}".format(heldout_lp))
      #
      #   qm_vals, qs_vals = sess.run((qmeans, qstds))
      #
      #   if HAS_SEABORN:
      #     plot_weight_posteriors(names, qm_vals, qs_vals,
      #                            fname=os.path.join(
      #                                FLAGS.model_dir,
      #                                "step{:05d}_weights.png".format(step)))
      #
      #     plot_heldout_prediction(image_vals, probs,
      #                             fname=os.path.join(
      #                                 FLAGS.model_dir,
      #                                 "step{:05d}_pred.png".format(step)),
      #                             title="mean heldout logprob {:.2f}"
      #                             .format(heldout_lp))

if __name__ == "__main__":
  tf.app.run()
