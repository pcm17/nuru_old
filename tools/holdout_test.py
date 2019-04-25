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

# Code for flags to choose which disease to run model on
parser = argparse.ArgumentParser()
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
#Choosing cut off for precision levels
parser.add_argument(
  '--precision_level',
  type=float,
  default=0.6,
  help="Choose precision threshold"
)

FLAGS, unparsed = parser.parse_known_args()

# Set strings to use for file paths based on the chosen disease + color for corresponding boxes for a specific disease
if FLAGS.disease_name == 'BSD':
  disease_abbrev = 'BSD/'
  disease_string = 'brown_streak/'
  disease_color = 'Brown'

elif FLAGS.disease_name == 'MD':
  disease_abbrev = 'MD/'
  disease_string = 'mosaic/'
  disease_color = 'Blue'

elif FLAGS.disease_name == 'GM':
  disease_abbrev = 'GM/'
  disease_string = 'green_mite/'
  disease_color = 'Chartreuse'

elif FLAGS.disease_name == 'RM':
  disease_abbrev = 'RM/'
  disease_string = 'red_mite/'
  disease_color = 'BlueViolet'

elif FLAGS.disease_name == 'BLS':
  disease_abbrev = 'BLS/'
  disease_string = 'brown_leaf_spot/'
  disease_color = 'Yellow'

elif FLAGS.disease_name == 'HL':
  disease_abbrev = 'HL/'
  disease_string = 'healthy/'
  disease_color = 'White'

elif FLAGS.disease_name == 'NUTD':
  disease_abbrev = 'NUTD/'
  disease_string = 'Nutrient_deficiency/'
  disease_color = 'Orchid'

model = FLAGS.model_name
precision_threshold = FLAGS.precision_level

# Checks box sizes 
def size_checker(sizes):
  size_val = 'Small/'
  for size in sizes:
    if size >= 93750:
      size_val = 'Big/'
  return size_val 

# Checks box colors to see if image is confused
def precision_calculator(real_color, boxes):
  total_boxes = len(boxes)
  correct_count = 0
  for box, color in boxes:
    if color == real_color:
      correct_count += 1
  if total_boxes == 0:
    return 0
  else: 
    return float(correct_count)/float(total_boxes)

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

# What model to download.
MODEL_NAME = '/Users/singhcpt/dev/object_detection/Cassava/model_test_working/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'model/pb_files/Cassava_detect_' + model + '.pb' # (insert crop name here)s_detect_model

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
PATH_TO_TEST_IMAGES_DIR = MODEL_NAME + 'images/leaflet_dataset/' + disease_string 
PATH_TO_OUTPUT_DIR =  MODEL_NAME + 'output_' + model + '/'
PATH_TO_RESULTS_IMAGES_DIR = MODEL_NAME + 'output_' + model + '/' + disease_abbrev 
CLASS = disease_abbrev # FLAG HERE

TEST_IMAGE_PATHS = [ PATH_TO_TEST_IMAGES_DIR + file_name for file_name in os.listdir(PATH_TO_TEST_IMAGES_DIR) ]
# Keep this for testing code over small portions of the folder, uses indices of images rather than entire directory. 
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'CassavaBLS' + '_{}.jpg'.format(i)) for i in range(25, 40) ] 

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Initializing counts that will be used in the results file
totalCount = 0
normalCount = 0
confusionCount = 0
precisions = []

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
      image = Image.open(image_path)
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

      # Checking box sizes, changing path based on it
      # SIZE_PATH = size_checker(boxes[2])

      # Checking whether image is confused and returning colors of confused boxes
      precision = precision_calculator(disease_color, boxes[1])
      precisions.append(precision)

      # Sorting into a folder according to precision threshold
      totalCount += 1
      if precision >= precision_threshold:
        img = Image.fromarray(image_np, 'RGB')
        img.save(PATH_TO_RESULTS_IMAGES_DIR + 'Normal/'+ file_name + "_output.jpg", "JPEG")
        normalCount += 1
        print("Successful detection: " + str(totalCount) + "/" + str(len(TEST_IMAGE_PATHS)) + " | Precision: " + str(precision) + " | Normal: " + str(normalCount) + " | Confused: " + str(confusionCount))
      else:
        img = Image.fromarray(image_np, 'RGB')
        img.save(PATH_TO_RESULTS_IMAGES_DIR + 'Confused/'+ file_name + "_output.jpg", "JPEG")
        confusionCount += 1
        print("Confused detection: " + str(totalCount) + "/" + str(len(TEST_IMAGE_PATHS)) + " | Precision: " + str(precision) + " | Normal: " + str(normalCount) + " | Confused: " + str(confusionCount))

  # Printing results into excel sheet    
  workbook = xlsxwriter.Workbook('/Users/singhcpt/dev/object_detection/Cassava/model_test_working/output_' + model + '/results' + disease_abbrev[:-1] + '.xlsx')
  worksheet = workbook.add_worksheet()
  
  worksheet.write('A1', 'Disease')
  worksheet.write('B1', 'Total Images')
  worksheet.write('C1', 'Normal Images')
  worksheet.write('D1', 'Confused Images')
  worksheet.write('E1', 'Error Rate')
  worksheet.write('F1', 'Average Precision')
  worksheet.write('A2', disease_abbrev[:-1])
  worksheet.write('B2', totalCount)
  worksheet.write('C2', normalCount)
  worksheet.write('D2', confusionCount)
  worksheet.write('E2', float(confusionCount)/float(totalCount))
  worksheet.write('F2', sum(precisions)/float(len(precisions)))
  #change precision to accuracy 
