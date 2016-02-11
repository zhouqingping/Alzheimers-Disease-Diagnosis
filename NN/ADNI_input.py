# Copyright 2015 Google Inc. All Rights Reserved.
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
# ==============================================================================
#
#
# This is a small scalable model built on Google's Convolutional Neural Network
#
#
# ==============================================================================


"""Routine for decoding .nii mri files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import deque as Q

import tensorflow as tf
import XMLReader
import numpy as np

from tensorflow.python.platform import gfile
from nifti import *


# Process images of this size.
# image size of 256 x 256 x 170
# If one alters this, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 256  # W and L (x and y)
IMAGE_SLICE = 170  # depth / slice (z)

# Global constants describing the ADNI data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10


def read_ADNI_image(filename_queue):
    """Reads NiFTi data file, next in the queue.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (256)
      width: number of columns in the result (256)
      slice: number of slices (3d mri) (170)
      key: file name
      label: an int32 Tensor with the label 0, 1, or 2.
      float32image: a [height, width, depth] float32 Tensor with the image data
      nim: nii image reader/writer
      rawImage: Numpy array
  """

    class mri(object):
        pass

    result = mri()
    im, r = filename_queue.popleft() # FIFO so popleft
    result.nim = NiftiImage(im)
    result.rawImage = result.nim.data # numpy array
    # padding to make sure the numpy array is the right shape
    dims = result.nim.extent
    xadd = 256-dims[0]
    if xadd < 0: xadd = 0
    yadd = 256-dims[1]
    if yadd < 0: yadd = 0
    zadd = 170-dims[2]
    if zadd < 0: zadd = 0
    result.rawImage = np.pad(result.rawImage, ((0, xadd), (0, yadd), (0, zadd)))
    result.float32image = tf.convert_to_tensor(result.rawImage)
    # Dimensions of the images in the ADNI dataset.
    result.height = 256
    result.width = 256
    result.slice = 170  # brain 'slice' of an MRI
    # file name for reference later
    result.filename = result.nim.filename

    result.label = r.subject_status()

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
    """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [width, height, slice] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, width, height, slice] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def inputs(eval_data, data_dir, batch_size):
    """Construct input

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, width, height, slice] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
    if not eval_data:
        all = os.listdir(data_dir)
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[len(temp)-1] == "nii":
                xmls.append(x)
        ims = []
        for f in xmls:
            r = XMLReader(f)
            p = r.path_to_scan(data_dir)
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(p)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        all = os.listdir(data_dir + "/testing_data")
        xmls = []
        for x in all:
            temp = x.split(".")
            if temp[1] == "nii":
                xmls.append(x)
        ims = []
        for f in xmls:
            r = XMLReader(f)
            p = r.path_to_scan(data_dir)
            if not gfile.Exists(p):
                raise ValueError('Failed to find file: ' + f)
            else:
                ims.append(p)
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    # Create a queue that produces the filenames to read.
    filename_queue = Q(iterable=ims, maxlen=len(ims))

    # Read examples from files in the filename queue.
    im = read_ADNI_image(filename_queue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(im.float32image, im.label,
                                           min_queue_examples, batch_size)
