from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import os
import librosa
import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
slim = tf.contrib.slim

_max_len = 8000

def mu_law(x, mu=255, int8=False):
  """A TF implementation of Mu-Law encoding.

  Args:
    x: The audio samples to encode.
    mu: The Mu to use in our Mu-Law.
    int8: Use int8 encoding.

  Returns:
    out: The Mu-Law encoded int8 data.
  """
  out = tf.sign(x) * tf.log(1 + mu * tf.abs(x)) / np.log(1 + mu)
  out = tf.floor(out * (mu / 2 + 1))
  if int8:
    out = tf.cast(out, tf.int8)
  return out


def inv_mu_law(x, mu=255):
  """A TF implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """
  x = tf.cast(x, tf.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = tf.sign(out) / mu * ((1 + mu)**tf.abs(out) - 1)
  out = tf.where(tf.equal(x, 0), x, out)
  return out


def inv_mu_law_numpy(x, mu=255.0):
  """A numpy implementation of inverse Mu-Law.

  Args:
    x: The Mu-Law samples to decode.
    mu: The Mu we used to encode these samples.

  Returns:
    out: The decoded data.
  """
  x = np.array(x).astype(np.float32)
  out = (x + 0.5) * 2. / (mu + 1)
  out = np.sign(out) / mu * ((1 + mu)**np.abs(out) - 1)
  out = np.where(np.equal(x, 0), x, out)
  return out



def causal_linear(x, n_inputs, n_outputs, name, filter_length, rate,
                  batch_size):
  """Applies dilated convolution using queues.

  Assumes a filter_length of 3.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.
    filter_length: The length of the convolution, assumed to be 3.
    rate: The rate or dilation
    batch_size: Non-symbolic value for batch_size.

  Returns:
    y: The output of the operation
    (init_1, init_2): Initialization operations for the queues
    (push_1, push_2): Push operations for the queues
  """
  assert filter_length == 3

  # create queue
  q_1 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
  q_2 = tf.FIFOQueue(rate, dtypes=tf.float32, shapes=(batch_size, 1, n_inputs))
  init_1 = q_1.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
  init_2 = q_2.enqueue_many(tf.zeros((rate, batch_size, 1, n_inputs)))
  state_1 = q_1.dequeue()
  push_1 = q_1.enqueue(x)
  state_2 = q_2.dequeue()
  push_2 = q_2.enqueue(state_1)

  # get pretrained weights
  w = tf.get_variable(
      name=name + "/W",
      shape=[1, filter_length, n_inputs, n_outputs],
      dtype=tf.float32)
  b = tf.get_variable(
      name=name + "/biases", shape=[n_outputs], dtype=tf.float32)
  w_q_2 = tf.slice(w, [0, 0, 0, 0], [-1, 1, -1, -1])
  w_q_1 = tf.slice(w, [0, 1, 0, 0], [-1, 1, -1, -1])
  w_x = tf.slice(w, [0, 2, 0, 0], [-1, 1, -1, -1])

  # perform op w/ cached states
  y = tf.nn.bias_add(
      tf.matmul(state_2[:, 0, :], w_q_2[0][0]) + tf.matmul(
          state_1[:, 0, :], w_q_1[0][0]) + tf.matmul(x[:, 0, :], w_x[0][0]), b)

  y = tf.expand_dims(y, 1)
  return y, (init_1, init_2), (push_1, push_2)


def linear(x, n_inputs, n_outputs, name):
  """Simple linear layer.

  Args:
    x: The [mb, time, channels] tensor input.
    n_inputs: The input number of channels.
    n_outputs: The output number of channels.
    name: The variable scope to provide to W and biases.

  Returns:
    y: The output of the operation.
  """
  w = tf.get_variable(
      name=name + "/W", shape=[1, 1, n_inputs, n_outputs], dtype=tf.float32)
  b = tf.get_variable(
      name=name + "/biases", shape=[n_outputs], dtype=tf.float32)
  y = tf.nn.bias_add(tf.matmul(x[:, 0, :], w[0][0]), b)
  y = tf.expand_dims(y, 1)
  return y

def int_shape(x):
    return list(map(int, x.get_shape()))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x - m), axis, keep_dims=True))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x - m2), axis))

def logistic_likelihood(param, audio, mu=65535):
    centered_x = audio - param[:,:,0:1]
    inv_stdv = tf.exp(-param[:,:,1:2])
    plus_in = inv_stdv * (centered_x + 1. / mu)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / mu)
    cdf_min = tf.nn.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min
    return cdf_delta


def discretized_mix_logistic_loss(x, l, nr_mix, mu=65535):

    logit_probs = l[:, :, :nr_mix]

    batch = x.get_shape()[0].value
    x = tf.reshape(x, [batch, -1, 1])
    x = tf.tile(x, (1, 1, nr_mix))
    means = l[:, :, nr_mix:2 * nr_mix]
    log_scales = l[:, :, 2 * nr_mix:]
    log_scales = tf.maximum(log_scales, -7.)

    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / mu)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / mu)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus(mid_in)
    log_probs = tf.where(cdf_delta < 1e-5, log_pdf_mid - np.log((mu+1)/2.), tf.log(tf.maximum(cdf_delta, 1e-5)))
    log_probs = log_probs + log_prob_from_logits(logit_probs)
    
    return -tf.reduce_sum(log_sum_exp(log_probs))
    

def sample_from_discretized_mix_logistic(l, nr_mix, mu=65535):
    logit_probs = l[:, :, :nr_mix]
    means = l[:, :, nr_mix:2 * nr_mix]
    log_scales = l[:, :, 2 * nr_mix:]
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform([1, 1, nr_mix],
                     minval=1e-5, maxval=1. - 1e-5))), axis=2), 
                     depth=nr_mix, dtype=tf.float32)
    means = tf.reduce_sum(means * sel, 2)
    log_scales = tf.maximum(tf.reduce_sum(log_scales * sel, 2), -7.)
    u = tf.random_uniform([1, 1], minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.log(u) - tf.log(1. - u))
    return tf.minimum(tf.maximum(x, -1.), 1.)


def sample_one_logistic_mix_param(l, nr_mix):
    logit_probs = l[:, :, :nr_mix]
    means = l[:, :, nr_mix:2 * nr_mix]
    log_scales = l[:, :, 2 * nr_mix:]
    sel = tf.one_hot(tf.argmax(logit_probs, axis=2), depth=nr_mix, dtype=tf.float32)
    means = tf.reduce_sum(means * sel, 2)
    log_scales = tf.maximum(tf.reduce_sum(log_scales * sel, 2), -12.)
    return tf.stack([means, log_scales], axis=-1)


