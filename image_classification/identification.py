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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from prettytable import PrettyTable

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import shutil
import itertools
import subprocess

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt; plt.rcdefaults()

def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.
  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def get_image_list(image_dir):
  if not tf.gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  # The root directory comes first, so skip it.
  sub_dirs = sorted(x[0] for x in tf.gfile.Walk(image_dir))
  imglist = []
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = sorted(set(os.path.normcase(ext)  # Smash case on Windows.
                            for ext in ['JPEG', 'JPG', 'jpeg', 'jpg', 'png', 'PNG']))
    file_list = []
    dir_name = image_dir
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(tf.gfile.Glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    for f in file_list:
      base_name = os.path.basename(f)
      print(base_name)
      imglist.append(base_name)
  return imglist

def save_images(folder, imgName, test_filename):
  image = Image.open(test_filename)
  ensure_dir_exists((FLAGS.results_file_directory + folder))
  os.chdir(FLAGS.results_file_directory + folder)
  image.save(FLAGS.results_file_directory + folder +imgName)
  image.close()
  subprocess.call(['sudo','chown', FLAGS.username, FLAGS.results_file_directory + folder])
  os.chdir(FLAGS.image_file_directory)

def read_tensor_from_image_file(file_name,
                                input_height,
                                input_width,
                                input_mean,
                                input_std):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(labels_list):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(FLAGS.labels).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  
  return label


if __name__ == "__main__":
  #Flags
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--image", 
      default = "/home/plantvillage/blurry_classification_dataset/nontrainingimages/images/578_cassava_bottom_3_capturedblurry.png",
      help="image to be processed"
  )
  parser.add_argument(
      "--username", 
      type=str,
      default="plantvillage",
      help="name of the user for permissions"
  )
  parser.add_argument(
      "--image_file_directory", 
      type=str,
      default="/home/plantvillage/blurry_classification_dataset/nontrainingimages/images/",
      help="name of image file directory to be processed"
  )
  parser.add_argument(
      "--results_file_directory", 
      type=str, 
      default="/home/plantvillage/blurry_classification_dataset/nontrainingimages/results/",
      help="name of the file directory to save the images to, with their labels"
  )
  parser.add_argument(
      "--graph", 
      type=str,
      default="/tmp/output_graph.pb",
      help="graph/model to be executed"
  )
  parser.add_argument(
      "--labels", 
      default="/tmp/output_labels.txt",
      help="labels to be used in classification"
  )
  parser.add_argument(
      "--input_height", 
      type=int,
      default=299, 
      help="input height"
  )
  parser.add_argument(
      "--input_width", 
      type=int, 
      default=299,
      help="input width"
  )
  parser.add_argument(
      "--input_mean", 
      type=int, 
      default=0,
      help="input mean"
  )
  parser.add_argument(
      "--input_std", 
      type=int, 
      default=255,
      help="input std"
  )
  parser.add_argument(
      "--input_layer", 
      default="Placeholder",
      help="name of input layer"
  )
  parser.add_argument(
      "--output_layer", 
      default="final_result",
      help="name of output layer"
  )
  FLAGS = parser.parse_args()

  image_lists = os.listdir(FLAGS.image_file_directory)

  graph = load_graph(FLAGS.graph)

  input_name = "import/" + FLAGS.input_layer
  output_name = "import/" + FLAGS.output_layer
  input_operation = graph.get_operation_by_name(input_name)
  output_operation = graph.get_operation_by_name(output_name)

  for fn in image_lists:
    filenameimg = "/home/plantvillage/blurry_classification_dataset/nontrainingimages/images/" + fn
    t = read_tensor_from_image_file(
        filenameimg,
        input_height=FLAGS.input_height,
        input_width=FLAGS.input_width,
        input_mean=FLAGS.input_mean,
        input_std=FLAGS.input_std)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    print("\n"+fn)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(FLAGS.labels)
    toplabel = str(labels[top_k[0]]).replace(" ", "") + "/"
    for i in top_k:
      print(labels[i], results[i])
    save_images(toplabel, fn, filenameimg)