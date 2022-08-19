# -*- coding: utf-8 -*-

import numpy as np
import imageio
import tensorflow as tf
from tensorflow.keras import backend as K


def read_pfm(fpath, expected_identifier="Pf"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' %
                            (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' %
                            line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception(
                'Could not parse max value / endianess information: "%s". '
                'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception(
                'Invalid binary values. Could not create %dx%d array from input.'
                % (height, width))

        return data


def load_LFdata(dir_LFimages):
    traindata_all = np.zeros((len(dir_LFimages), 512, 512, 9, 9, 3), np.uint8)
    traindata_label = np.zeros((len(dir_LFimages), 512, 512), np.float32)

    image_id = 0
    for dir_LFimage in dir_LFimages:
        print(dir_LFimage)
        for i in range(81):
            try:
                tmp = np.float32(
                    imageio.imread('hci_dataset/' + dir_LFimage +
                                   '/input_Cam0%.2d.png' %
                                   i))  # load LF images(9x9)
            except:
                print('hci_dataset/' + dir_LFimage +
                      '/input_Cam0%.2d.png..does not exist' % i)
            traindata_all[image_id, :, :, i // 9, i - 9 * (i // 9), :] = tmp
            del tmp
        try:
            tmp = np.float32(
                read_pfm('hci_dataset/' + dir_LFimage +
                         '/gt_disp_lowres.pfm'))  # load LF disparity map
        except:
            print('hci_dataset/' + dir_LFimage +
                  '/gt_disp_lowres.pfm..does not exist' % i)
        traindata_label[image_id, :, :] = tmp
        del tmp
        image_id = image_id + 1
    return traindata_all, traindata_label


'''
 ' Huber loss.
 ' https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
 ' https://en.wikipedia.org/wiki/Huber_loss
'''

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


def huber(y_true, y_pred, delta=1.0):
    """Computes Huber loss value.
  For each value x in `error = y_true - y_pred`:
  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = d * |x| - 0.5 * d^2        if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss
  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.
  Returns:
    Tensor with one scalar loss entry per sample.
  """
    y_pred = math_ops.cast(y_pred, dtype=K.floatx())
    y_true = math_ops.cast(y_true, dtype=K.floatx())
    delta = math_ops.cast(delta, dtype=K.floatx())
    error = math_ops.subtract(y_pred, y_true)
    abs_error = math_ops.abs(error)
    half = ops.convert_to_tensor_with_dispatch(0.5, dtype=abs_error.dtype)
    return K.mean(array_ops.where(
        abs_error <= delta, half * math_ops.square(error),
        delta * abs_error - half * math_ops.square(delta)),
                  axis=-1)


def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.keras.backend.mean(tf.where(cond, squared_loss, linear_loss))


def huber_loss2(y_true, y_pred, delta=1.0):
    error = math_ops.subtract(y_pred, y_true)
    abs_error = math_ops.abs(error)
    half = ops.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return K.mean(array_ops.where(
        abs_error <= delta, half * math_ops.square(error),
        delta * abs_error - half * math_ops.square(delta)),
                  axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = math_ops.abs(
        (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)
