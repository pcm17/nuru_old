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


import hashlib
import io
import logging
import os
import random
import re
import glob
import cv2
import shutil
from lxml import etree
import PIL
import tensorflow as tf
import numpy as np

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/validation_images/23/xmls', 'Path to directory to output TFRecords.')
flags.DEFINE_string('trainval_path','/home/plantvillage/Dropbox/Object_Detection/spore/data/annotations/trainval_23.txt','Path to trainval txt file with list of files')
FLAGS = flags.FLAGS

def main(_):
  data_dir = FLAGS.data_dir
  output_dir = FLAGS.output_dir
  logging.info('Reading from the dataset.')
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_path = FLAGS.trainval_path
  examples_list = dataset_util.read_examples_list(examples_path)

  indices = [ i for i, word in enumerate(examples_list) if (word.startswith('spore-50') or word.startswith('spore-51')) ]
  result_list = [examples_list[i] for i in indices]   
  #print(result_list)
  for i in enumerate(reversed(indices)):
    del examples_list[i[1]]

  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.8 * num_examples)
  val_examples = examples_list[num_train:]
  print(val_examples)
  print('Extracting validation images')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for file_name in val_examples:
    src_full_path = data_dir + 'annotations/xmls/' + file_name + '.xml'
    try:
			open(src_full_path)
    except IOError:		
      print('xml with the following path doesn\'t exist:' + src_full_path)
      continue
    out_path = os.path.join(output_dir,(file_name + '.xml'))
    shutil.copy(src_full_path, out_path)

  
  print('Complete!')

if __name__ == '__main__':
  tf.app.run()
