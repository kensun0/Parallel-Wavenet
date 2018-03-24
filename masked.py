from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def shift_right(x):
  """Shift the input over by one and a zero to the front.

  Args:
    x: The [mb, time, channels] tensor input.

  Returns:
    x_sliced: The [mb, time, channels] tensor output.
  """
  shape = tf.shape(x)
  x_padded = tf.pad(x, [[0, 0], [1, 0], [0, 0]])
  x_sliced = tf.slice(x_padded, [0, 0, 0], tf.stack([-1, shape[1], -1]))
  x_sliced=tf.reshape(x_sliced, shape)
  return x_sliced


def conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(0.0001),
           biases_initializer=tf.constant_initializer(0.0),
           trainable=True):
  """Fast 1D convolution that supports causal padding and dilation.

  Args:
    x: The [mb, time, channels] float tensor that we convolve.
    num_filters: The number of filter maps in the convolution.
    filter_length: The integer length of the filter.
    name: The name of the scope for the variables.
    dilation: The amount of dilation.
    causal: Whether or not this is a causal convolution.
    kernel_initializer: The kernel initialization function.
    biases_initializer: The biases initialization function.

  Returns:
    y: The output of the 1D convolution.
  """
  batch_size, length, num_input_channels = x.get_shape().as_list()
  batch_size = tf.shape(x)[0]
  length = tf.shape(x)[1]

  kernel_shape = [1, filter_length, num_input_channels, num_filters]
  strides = 1
  biases_shape = [num_filters]
  padding = 'VALID' if causal else 'SAME'

  
  with tf.variable_scope(name):
    weights = tf.get_variable(
        'W', shape=kernel_shape, initializer=kernel_initializer)
    biases = tf.get_variable(
        'biases', shape=biases_shape, initializer=biases_initializer)
 
  
  kernel=tf.reshape(weights,[filter_length, num_input_channels, num_filters])
  
  if causal:
    left_pad = dilation * (kernel_shape[1] - 1)
    x = temporal_padding(x, (left_pad, 0))
    padding = 'VALID'
  else:
    padding = 'SAME'

  tf_data_format = 'NWC'
  y = tf.nn.convolution(
        input=x,
        filter=kernel,
        dilation_rate=(dilation,),
        strides=(strides,),
        padding=padding,
        data_format=tf_data_format)
  y=tf.reshape(y,[tf.shape(y)[0], 1, tf.shape(y)[1], num_filters])
  y=tf.nn.bias_add(y,biases)
  y=tf.reshape(y,[tf.shape(y)[0], tf.shape(y)[2], num_filters])
  return y

def temporal_padding(x, padding=(1, 1)):
  """Pads the middle dimension of a 3D tensor.
  # Arguments
      x: Tensor or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.
  # Returns
      A padded 3D tensor.
  """
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return tf.pad(x, pattern)
