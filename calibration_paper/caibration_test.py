# This file is a modification of holdout_test.py
# It strips confidences from each box and prints the confidence intervals against accuracy to an excel sheet

import argparse
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import xlsxwriter
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
# Choosing disease name 
parser.add_argument(
    '--disease_name',
    type=str,
    default='BSD',
    help='1 - BSD, 2 - MD, 3 - GM, 4 - RM, 5 - BLS, 6- HL, 7 - NUTD'
    )

# Choosing model type
parser.add_argument(
  '--model_name',
  type=str,
  default="",
  help='Write model name here'
)

FLAGS, unparsed = parser.parse_known_args()

# Set strings to use for file paths based on the chosen disease + color for corresponding boxes for a specific disease
if FLAGS.disease_name == 'BSD':
  disease_abbrev = 'BSD'
  disease_string = 'brown_streak/'
  disease_color = 'Brown'

elif FLAGS.disease_name == 'MD':
  disease_abbrev = 'MD'
  disease_string = 'mosaic/'
  disease_color = 'Blue'

elif FLAGS.disease_name == 'GM':
  disease_abbrev = 'GM'
  disease_string = 'green_mite/'
  disease_color = 'Chartreuse'

elif FLAGS.disease_name == 'RM':
  disease_abbrev = 'RM'
  disease_string = 'red_mite/'
  disease_color = 'BlueViolet'

elif FLAGS.disease_name == 'BLS':
  disease_abbrev = 'BLS'
  disease_string = 'brown_leaf_spot/'
  disease_color = 'Yellow'

elif FLAGS.disease_name == 'HL':
  disease_abbrev = 'HL'
  disease_string = 'healthy/'
  disease_color = 'White'

elif FLAGS.disease_name == 'NUTD':
  disease_abbrev = 'NUTD'
  disease_string = 'Nutrient_deficiency/'
  disease_color = 'Orchid'

model = FLAGS.model_name

FLAGS, unparsed = parser.parse_known_args()

model = FLAGS.model_name

# Converts color to correpsonding disease to adjust path to save file
def color_convert(color):
  if color == 'White':
    return 'HL' 
  elif color == 'Brown':
    return 'BSD'
  elif color == 'BlueViolet':
    return 'RM'
  elif color == 'Blue':
    return 'MD'
  elif color == 'Chartreuse':
    return 'GM'
  elif color == 'Yellow':
    return 'BLS'
  elif color == 'Orchid':
    return 'NUTD'

if tf.__version__ != '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

######## Environment Setup ########

# This is needed to display the images.
#matplotlib inline
# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")

######## Model Preparation ########

# What model to use.
MODEL_NAME = '/Users/singhcpt/dev/object_detection/Cassava/model_test_working/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'model/pb_files/Cassava_detect_' + model + '.pb' # (insert crop name here)_detect_model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'object-detection.pbtxt') #might need to change this, check data
NUM_CLASSES = 5

####### Load a (frozen) Tensorflow model into memory #######

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

######## Loading Label Map ########

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

######## Helper Code ########

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

######## Detection ########

# Directories adjusted based on the disease strings and model names (retrieved from flags)
PATH_TO_TEST_IMAGES_DIR = MODEL_NAME + 'images/leaflet_dataset/' + disease_abbrev + '/'
PATH_TO_OUTPUT_DIR =  MODEL_NAME + 'output_' + model + '/' + disease_abbrev + '/'
PATH_TO_RESULTS_IMAGES_DIR = MODEL_NAME + 'output_' + model + '/' 
CLASS = disease_abbrev # FLAG HERE

TEST_IMAGE_PATHS = [ PATH_TO_TEST_IMAGES_DIR + file_name for file_name in os.listdir(PATH_TO_TEST_IMAGES_DIR) ]
# Keep this for testing code over small portions of the folder, uses indices of images rather than entire directory. 
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Cassava' + disease_abbev + '_{}.jpg'.format(i)) for i in range(40, 80) ] 

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Declaring counts for confidence interval ranges
# 1 - 10% are at index 0, 11-20 % at index 1, etc. 
counts = []
for x in range(0, 10):
    counts.append([0, 0]) # First index of nested array is the total number of boxes, second is the correct number of boxes on the image

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    totalCount = 0
    normalCount = 0
    confusionCount = 0
    for image_path in TEST_IMAGE_PATHS:
      file_name, ext = os.path.splitext(os.path.basename(image_path))
      try:
        image = Image.open(image_path)
      except IOError:
        print('Found .DS_store at ' + image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 1]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection, boxes variable is an array with data on the boxes stored in it (stores their sizes, colors, confidences ,etc.) 
      boxes = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          image.size,
          use_normalized_coordinates=True,
          line_thickness=8)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      [[4, 3], [0, 0], [0, 0], [0, 0], [0, 0]]
      # Looping through boxes and respective confidences - boxes[1] is a 2D array with coordinates and colors, boxes[3] is an array of the respective confidence levels for each box
      for (box, confidence) in zip(boxes[1], boxes[3]):
          if 0 < confidence <= 10:
               counts[0][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[0][1] += 1
          if 10 < confidence <= 20:
               counts[1][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[1][1] += 1
          if 20 < confidence <= 30:
               counts[2][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[2][1] += 1
          if 30 < confidence <= 40:
               counts[3][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[3][1] += 1
          if 40 < confidence <= 50:
               counts[4][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[4][1] += 1
          if 50 < confidence <= 60:
               counts[5][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[5][1] += 1
          if 60 < confidence <= 70:
               counts[6][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[6][1] += 1
          if 70 < confidence <= 80:
               counts[7][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[7][1] += 1
          if 80 < confidence <= 90:
               counts[8][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[8][1] += 1
          if 90 < confidence <= 100:
               counts[9][0] += 1
               if color_convert(box[1]) in image_path:
                 counts[9][1] += 1
      
      img = Image.fromarray(image_np, 'RGB')
      img.save(PATH_TO_RESULTS_IMAGES_DIR + file_name + "_output.jpg", "JPEG")

      print(counts)

    # Printing results into excel sheet
    workbook = xlsxwriter.Workbook('/Users/singhcpt/dev/object_detection/Cassava/model_test_working/output_' + model + '/results.xlsx')
    worksheet = workbook.add_worksheet()

    worksehet.write('A1', disease_abbrev)

    worksheet.write('A2', 'Confidence Level (%)')
    worksheet.write('B2', '1 - 10')
    worksheet.write('C2', '11 - 20')
    worksheet.write('D2', '21 - 30')
    worksheet.write('E2', '31 - 40')
    worksheet.write('F2', '41 - 50')
    worksheet.write('G2', '51 - 60')
    worksheet.write('H2', '61 - 70')
    worksheet.write('I2', '71 - 80')
    worksheet.write('J2', '81 - 90')
    worksheet.write('K2', '91 - 100')

    worksheet.write('A3', 'Accuracy')
    worksheet.write('B3', float(counts[0][1])/float(counts[0][0]) if counts[0][0] != 0 else 0)
    worksheet.write('C3', float(counts[1][1])/float(counts[1][0]) if counts[1][0] != 0 else 0)
    worksheet.write('D3', float(counts[2][1])/float(counts[2][0]) if counts[2][0] != 0 else 0)
    worksheet.write('E3', float(counts[3][1])/float(counts[3][0]) if counts[3][0] != 0 else 0)
    worksheet.write('F3', float(counts[4][1])/float(counts[4][0]) if counts[4][0] != 0 else 0)
    worksheet.write('G3', float(counts[5][1])/float(counts[5][0]) if counts[5][0] != 0 else 0)
    worksheet.write('H3', float(counts[6][1])/float(counts[6][0]) if counts[6][0] != 0 else 0)
    worksheet.write('I3', float(counts[7][1])/float(counts[7][0]) if counts[7][0] != 0 else 0)
    worksheet.write('J3', float(counts[8][1])/float(counts[8][0]) if counts[8][0] != 0 else 0)
    worksheet.write('K3', float(counts[9][1])/float(counts[9][0]) if counts[9][0] != 0 else 0)

    worksheet.write('A4', 'Counts')
    worksheet.wriet('B4', counts[0][1] + '/' + counts[0][0]) 
    worksheet.wriet('C4', counts[1][1] + '/' + counts[1][0])
    worksheet.wriet('D4', counts[2][1] + '/' + counts[2][0])
    worksheet.wriet('E4', counts[3][1] + '/' + counts[3][0])
    worksheet.wriet('F4', counts[4][1] + '/' + counts[4][0])
    worksheet.wriet('G4', counts[5][1] + '/' + counts[5][0])
    worksheet.wriet('H4', counts[6][1] + '/' + counts[6][0])
    worksheet.wriet('I4', counts[7][1] + '/' + counts[7][0])
    worksheet.wriet('J4', counts[8][1] + '/' + counts[8][0])
    worksheet.wriet('K4', counts[9][1] + '/' + counts[9][0])