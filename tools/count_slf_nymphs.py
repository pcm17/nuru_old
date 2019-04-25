'''
python object_detection/count_slf_nymphs.py \
--image_dir /home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/val_test \
--output_dir /home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/val_output \
--model_path /home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/model/6x4_section_dataset/output_pb_files/slf_blackinstar_v0.8.pb \
--labels_path /home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/blackinstar_label_map.pbtxt \
--gt_file /home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/gt_files_counts.txt \
--threshold=0.6
'''
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

flags = tf.app.flags
flags.DEFINE_string('image_dir', '/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/val_test2', 'Path to the parent directory of the project')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/val_output2', 'Path to the parent directory of the project')
flags.DEFINE_string('model_path','/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/model/6x4_section_dataset/output_pb_files/slf_blackinstar_v0.8.pb2','The full file path to the model')
flags.DEFINE_string('labels_path','/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/blackinstar_label_map.pbtxt2','The full file path to the labels file')
flags.DEFINE_string('gt_file','/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/data/6x4_section_dataset/gt_files_counts.txt2', 'The text file containing the filenumbers and groundtruth counts')
flags.DEFINE_float('threshold', 0.5, 'The threshold for detection confidences')
FLAGS = flags.FLAGS

######## Model Preparation ########
PATH_TO_TEST_IMAGES_DIR = FLAGS.image_dir
PATH_TO_RESULTS_IMAGES_DIR = FLAGS.output_dir
PATH_TO_MODEL = FLAGS.model_path
DATA_DIR = FLAGS.image_dir
MODEL_NAME = os.path.basename(PATH_TO_MODEL)

MODEL_VERSION = MODEL_NAME[(len(MODEL_NAME)-6):(len(MODEL_NAME)-3)]


PATH_TO_LABELS = FLAGS.labels_path
score_threshold = FLAGS.threshold
gt_file_path = FLAGS.gt_file
NUM_CLASSES = 1

# Check that gt counts file and data_dir have the same number of items



######## Initialize Data Array ########
gt_data = file_data = np.genfromtxt(gt_file_path, usecols=(0,1), skip_header=0, dtype=str)
if np.size(gt_data,0) != len(os.listdir(PATH_TO_TEST_IMAGES_DIR)):
  print('\n\tNum rows in gt_counts.txt must be EQUAL to number of files in image_dir\n\n\tENDING SCRIPT')
  sys.exit()

# Concatenate a column onto the data array for predicted counts
num_data_pts = np.size(gt_data,0)
data = np.zeros((num_data_pts, 3))
data[:,:-1] = gt_data


####### Load a (frozen) Tensorflow model into memory #######
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
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

def count_boxes(boxes):
  return len(boxes)


# If path to results directory doesn't exist, create the directory
if not os.path.exists(PATH_TO_RESULTS_IMAGES_DIR):
  os.makedirs(PATH_TO_RESULTS_IMAGES_DIR)
  print('\n\tCreating new output folder\n')

print('\n\n\tCounting has started!\n')

######## Detection ########
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
      # Check that there is gt data for this filenum
      file_num = file_name[(len(file_name)-3):(len(file_name))]
      img_row = np.where(data == int(file_num))[0]
      if img_row.size == 0:
        print('\n\tNo groundtruth data found in gt_counts.txt\n\n\tENDING SCRIPT')
        sys.exit()
      image = Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,image_path))
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
      boxes = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          image.size,
          use_normalized_coordinates=True,
          max_boxes_to_draw=None,
          min_score_thresh=score_threshold,
          line_thickness=6)

      pr_count = count_boxes(boxes[1])
      # Find the corresponding image number and update the gt count
      gt_count = int(data[int(img_row),1])   

      # Define output counts file and create it if it doesn't exist
      print('Image: %s \t GT Count:%s \t Predicted Count:%d' % (file_name, gt_count, pr_count))
      counts_filename = os.path.join(PATH_TO_RESULTS_IMAGES_DIR, ('SLF_counts_' + 'v' + MODEL_VERSION + '_' + str(score_threshold) + '%.txt'))
      if os.path.exists(counts_filename):
        append_write = 'a' # append if already exists
      else:
        append_write = 'w' # make a new file if not
      # Write the data to the file
      file_ = open(counts_filename, append_write)
      file_.write('%s\t%d\t\t\t%d\n' % (file_num, gt_count, pr_count))
      file_.close()
      # Save image overlayed with bounding boxes
      img = Image.fromarray(image_np, 'RGB')
      print('Saving image to %s' % str(PATH_TO_RESULTS_IMAGES_DIR + '/' + file_name + "_output.jpg"))
      img.save(os.path.join(PATH_TO_RESULTS_IMAGES_DIR, (file_name + "_output.jpg")), "JPEG")
