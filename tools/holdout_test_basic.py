import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

######## Flag Definitions ########
flags = tf.app.flags
flags.DEFINE_string('project_name', '', 'The name of the project that this test is being ran for')
flags.DEFINE_string('project_dir', '', 'The path to the root directory for the project that this test is being ran for')
flags.DEFINE_string('image_dir', '', 'Relative path from project directory to holdout images. ')
flags.DEFINE_string('model_dir', '', 'Relative path from project directory to model pb file.') 
flags.DEFINE_string('version', '', 'The model version number')
FLAGS = flags.FLAGS

######## Model Preparation ########

# What model to load.
if FLAGS.project_name == 'spores':
  MODEL_NAME = 'spores_detect_v'
  NUM_CLASSES = 1
elif FLAGS.project_name == 'slf':
  MODEL_NAME = 'slf_blackinstar_detect_v'	## TODO: Make sure this is the correct model name prefix
  NUM_CLASSES = 1
elif FLAGS.project_name == 'cassava':
  MODEL_NAME = 'cassava_detect_v'
  NUM_CLASSES = 7
elif FLAGS.project_name == 'faw':
  MODEL_NAME = 'faw_detect_v'
  NUM_CLASSES = 2
elif FLAGS.project_name == 'wheat':
  MODEL_NAME = 'wheat_detect_v'
  NUM_CLASSES = 2
elif FLAGS.project_name == 'potato':
  MODEL_NAME = 'potato_detect_v'
  NUM_CLASSES = 2

PROJECT_DIR = FLAGS.project_dir
MODEL_DIR = PROJECT_DIR + FLAGS.model_dir
MODEL_FILE = MODEL_NAME + FLAGS.version + '.pb'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + 'cassava_combo_same_500.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'object-detection.pbtxt')


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

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = MODEL_NAME + 'images/'
PATH_TO_RESULTS_IMAGES_DIR = MODEL_NAME + 'output/'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'CassavaCMD_{}.jpg'.format(i)) for i in range(9, 13) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

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
    for image_path in os.listdir(PATH_TO_TEST_IMAGES_DIR):
      file_name, ext = os.path.splitext(os.path.basename(image_path))
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      #plt.figure(figsize=IMAGE_SIZE)
      #plt.imshow(image_np)
      img = Image.fromarray(image_np, 'RGB')
      img.save(PATH_TO_RESULTS_IMAGES_DIR + file_name + "_output.jpg", "JPEG")
