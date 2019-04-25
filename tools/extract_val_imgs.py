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
import glob
import cv2

from lxml import etree
import PIL
import tensorflow as tf
import numpy as np

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/validation_images/2/2_originals', 'Path to directory to output TFRecords.')
flags.DEFINE_string('trainval_path','/home/plantvillage/Dropbox/Object_Detection/spore/data/annotations/trainval.txt','Path to trainval txt file with list of files')
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
  print(result_list)
  for i in enumerate(reversed(indices)):
    del examples_list[i[1]]
  
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.8 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  train_examples += result_list
  print('Extracting validation images')
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for img_name in val_examples:
    full_path = data_dir + 'images/' + img_name + '.jpg'
    try:
			open(full_path)
    except IOError:		
      try:	
        full_path = data_dir + 'images/' + img_name + '.JPG'
        open(full_path)
        print('Not lowercase .jpg lets try UPPERCASE .JPG!!\nFile path:' + full_path)
      except IOError:
        try:
          full_path = data_dir + 'images/' + img_name + '.png'
          open(full_path)
          print('Not a .jpg lets try .png!!\nFile path:' + full_path)
        except IOError:
          print('NO IMAGE WITH THIS NAME: ' + full_path)
          continue

    with tf.gfile.GFile(full_path, 'rb') as fid:
      encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    out_path = os.path.join(output_dir,(img_name + '.JPG'))
    image.save(out_path)
    #txt_file.write("%s\n" % full_x)
    
  print('Complete!')

if __name__ == '__main__':
  tf.app.run()
