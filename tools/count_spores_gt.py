# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet 
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/validation_images/23/xmls', 'Root directory to raw pet dataset.')
flags.DEFINE_string('counts_file_path', '/home/plantvillage/Dropbox/Object_Detection/spore/counts/counts_23_val.txt', 'Path to directory to output TFRecords.')
flags.DEFINE_string('trainval_path', '/home/plantvillage/Dropbox/Object_Detection/spore/data/annotations/trainval_23.txt', 'Path to trainval text file')
FLAGS = flags.FLAGS


def count_spores(data):
  """
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    
  Returns:
    count: The ground truth count for spores in the image.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  print('Image: ' + data['filename'])
  count = 0
  for obj in data['object']:
    count += 1

  return count


def read_data(annotations_dir, examples, counts_filename):
  """Creates a TFRecord file from examples.

  Args:
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
  """
  if os.path.exists(counts_filename):
    append_write = 'a' # append if already exists
  else:
    append_write = 'w' # make a new file if not
  # Write the data to the file
  file_ = open(counts_filename, append_write)
  
  
  for idx, example in enumerate(sorted(examples)):
    print('On image %d out of %d' % (idx, len(examples)))
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    path = os.path.join(annotations_dir, example + '.xml')
    if not os.path.exists(path):
      #print('Could not find '+ os.path.basename(path))
      #logging.warning('Could not find %s, ignoring example.', path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    count = count_spores(data)
    file_.write('%s\t%d\n' % (example, count))
    print(count)
  
  file_.close()  
    

# TODO: Add test for pet/PASCAL main files.
def main(_):

  annotations_dir = FLAGS.data_dir
  examples_path = FLAGS.trainval_path
  examples_list = dataset_util.read_examples_list(examples_path)
  indices = [ i for i, word in enumerate(examples_list) if (word.startswith('spore-50') or word.startswith('spore-51')) ]
  result_list = [examples_list[i] for i in indices]   
  #print(result_list)
  for i in enumerate(reversed(indices)):
    del examples_list[i[1]]
    
  counts_filename = FLAGS.counts_file_path
  read_data(annotations_dir, examples_list, counts_filename)

  print('Complete!')

if __name__ == '__main__':
  tf.app.run()
