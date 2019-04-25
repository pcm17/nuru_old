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
    python tools/pad_boxes.py --data_dir=/home/plantvillage/Dropbox/Object_Detection/spore/data 
        --output_dir=/home/plantvillage/Dropbox/Object_Detection/spore/
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
flags.DEFINE_string('data_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/annotations/xmls', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/pad_working_output', 'Path to directory to output TFRecords.')
flags.DEFINE_integer('pad_size', 5, 'Number of pixels to add to pad')
FLAGS = flags.FLAGS



def pad_boxes(data, path, object_name, pad_size):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:

  Returns:

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  print('Adjusting boxes on image: ' + data['filename'])
  width = int(data['size']['width'])
  height = int(data['size']['height'])
  image_depth = 3

  print('Creating new xml at ' + path)
  f = open(path, 'w+')
  # Write the filename, file path 
  f.write('<annotation>\n\t<folder>DNA</folder>\n')
  f.write('\t<filename>%s</filename>\n' % (data['filename']))
  f.write('\t<path>DNA</path>\n')
  f.write('\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n\t<size>\n')
  f.write('\t\t<width>%s</width>\n\t\t<height>%s</height>\n\t\t<depth>%s</depth>\n' % (width, height, image_depth))
  f.write('\t</size>\n\t<segmented>0</segmented>\n')

  for obj in data['object']:
    
    xmin = int(obj['bndbox']['xmin'])
    ymin = int(obj['bndbox']['ymin'])
    xmax = int(obj['bndbox']['xmax'])
    ymax = int(obj['bndbox']['ymax'])
    xmin = xmin-pad_size
    ymin = ymin-pad_size
    xmax = xmax+pad_size
    ymax = ymax+pad_size
    f.write('\t<object>\n')
    f.write('\t\t<name>%s</name>\n' % (object_name))
    f.write('\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n')
    f.write('\t\t\t<xmin>%s</xmin>\n\t\t\t<ymin>%s</ymin>\n\t\t\t<xmax>%s</xmax>\n\t\t\t<ymax>%s</ymax>\n\t\t</bndbox>\n\t</object>\n' % (xmin,ymin,xmax,ymax))
  f.write('</annotation>')
  f.close()




# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  output_dir = FLAGS.output_dir
  pad_size = FLAGS.pad_size
  object_name = "spore"

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  for path in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, path)
    new_path = os.path.join(output_dir, os.path.basename(path))
    #print(new_path)
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    pad_boxes(data, new_path, object_name, pad_size)
  print('Complete!')

if __name__ == '__main__':
  tf.app.run()
