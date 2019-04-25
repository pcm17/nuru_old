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
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


######## Flag Definitions ########
flags = tf.app.flags

flags.DEFINE_float('threshold', 0.5, 'The threshold for detection confidences')
flags.DEFINE_string('record_version', '21', 'Path to the directory of images')
flags.DEFINE_string('model_version','21.0','Path to the model to load')
flags.DEFINE_string('crop', 'spore', 'Path to the label map text file')
FLAGS = flags.FLAGS
######## Flag Variable Preparation ########
crop = FLAGS.crop
model_version = FLAGS.model_version
record_version = FLAGS.record_version
score_threshold = FLAGS.threshold

PATH_TO_MODEL = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/model/pb_files/' + crop + '_detect_' + model_version + '.pb'
# Check that model exists
try:
	open(PATH_TO_MODEL)
except IOError:
  print("\n\nThere is no model with the following name:")
  print(PATH_TO_MODEL)
  sys.exit()
print("\nSuccessfully loaded model")
IMAGE_DIR_PATH = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/data/validation_images/' + record_version + '/' + record_version + '_originals/'
# Check that validation images exist
if not os.path.exists(IMAGE_DIR_PATH):
  print("\n\nThere is no validation image directory with the following name")
  print(IMAGE_DIR_PATH)
  sys.exit()
print("\nSuccessfully loaded images")
PATH_TO_LABELS = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/data/' + crop + '_label_map.pbtxt'
# Check that label map exists
try:
	open(PATH_TO_LABELS)
except IOError:
  print("\n\nThere is no label map with that name")
  sys.exit()
print("\nSuccessfully loaded label map")
PATH_TO_LABELS_LIST = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/data/' + crop + '_label_list.txt'
# Check that labels list exists
try:
	open(PATH_TO_LABELS_LIST)
except IOError:
  print("\n\nThere is no label list file with that name")
  sys.exit()
print("\nSuccessfully loaded labels list\n\n")

RESULTS_IMAGES_DIR_PATH = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/data/validation_images/' + record_version + '/' + model_version + '_results_' + str(score_threshold)
# If output directory does not already exist, create it
if not os.path.exists(RESULTS_IMAGES_DIR_PATH):
  os.makedirs(RESULTS_IMAGES_DIR_PATH)
  print("\nCreating new output directory")

if crop == 'spore' or crop == 'palm_tree_crown' or 'bee':
  ln_thickness = 4
else:
  ln_thickness = 10

if crop is 'Cassava':
  INCORRECT_OUTPUT_PATH = os.path.join(RESULTS_IMAGES_DIR_PATH, "incorrect") 
  CORRECT_OUTPUT_PATH = os.path.join(RESULTS_IMAGES_DIR_PATH, "correct")
  TIE_OUTPUT_PATH  = os.path.join(RESULTS_IMAGES_DIR_PATH, "tie")
  MIXED_CORRECT_OUTPUT_PATH  = os.path.join(RESULTS_IMAGES_DIR_PATH, "mixed_correct")

  # If output directory does not already exist, create it
  if not os.path.exists(INCORRECT_OUTPUT_PATH):
    os.makedirs(INCORRECT_OUTPUT_PATH)
  if not os.path.exists(CORRECT_OUTPUT_PATH):
    os.makedirs(CORRECT_OUTPUT_PATH)
  if not os.path.exists(TIE_OUTPUT_PATH):
    os.makedirs(TIE_OUTPUT_PATH)
  if not os.path.exists(MIXED_CORRECT_OUTPUT_PATH):
    os.makedirs(MIXED_CORRECT_OUTPUT_PATH)

if crop == 'potato' or crop is 'banana':
  INCORRECT_OUTPUT_PATH = os.path.join(RESULTS_IMAGES_DIR_PATH, "incorrect") 
  CORRECT_OUTPUT_PATH = os.path.join(RESULTS_IMAGES_DIR_PATH, "correct")
  
  # If output directory does not already exist, create it
  if not os.path.exists(INCORRECT_OUTPUT_PATH):
    os.makedirs(INCORRECT_OUTPUT_PATH)
  if not os.path.exists(CORRECT_OUTPUT_PATH):
    os.makedirs(CORRECT_OUTPUT_PATH)

# Class info Prep
with open(PATH_TO_LABELS_LIST) as f:
    gt_classes = f.read().splitlines()
num_classes = len(gt_classes) - 1
colors = vis_util.STANDARD_COLORS
gt_colors = colors[1:num_classes+1]
gt_classes_colors = np.append([gt_classes[1:]],[gt_colors], axis=0)
#print(gt_classes_colors)
accuracies = np.zeros((3, num_classes))

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
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Checks box colors to see if image is confused
def precision_calculator(gt_color, predicted_boxes):
  #print(predicted_boxes)
  total_boxes = len(predicted_boxes)
  correct_count = 0
  for box, color in predicted_boxes:
    if color == gt_color:
      correct_count += 1
  if total_boxes == 0:
    return 0
  else: 
    return float(correct_count)/float(total_boxes)

def make_diagnosis(gt_color, class_id, predicted_boxes):
  #print(predicted_boxes)
  total_boxes = len(predicted_boxes)
  # Tally up detections for each class
  predictions = np.zeros((1,len(gt_colors)), np.int8)
  
  for box, color in predicted_boxes:
    #print(color)
    if color is 'Red':
      predictions[0,0]=  predictions[0,0] + 1
    if color is 'Cyan':
      predictions[0,1] =  predictions[0,1] + 1
    if color is 'Green':
      predictions[0,2]=  predictions[0,2] + 1
    if color is 'Magenta':
      predictions[0,3]=  predictions[0,3] + 1
    if color is 'Yellow':
      predictions[0,4]=  predictions[0,4] + 1
    if color is 'White':
      predictions[0,5]=  predictions[0,5] + 1
    if color is 'BlueViolet':
      predictions[0,6]=  predictions[0,6] + 1
  
  prediction = np.argmax(predictions)
  
  winner = np.argwhere(predictions == np.amax(predictions))
  #print(predictions)
  #print(len(winner))
  #print('\nPrediction: ' + str(winner) + ' GT: ' + str(class_id))
  if len(winner) == 1:
    if prediction == class_id:
      return 1
    else:
      return 0
  else:
    return 2
  
    
# Extract the class name from the filename and match it to the corresponding box color
def get_class_color(gt_class_name, gt_classes_colors, num_classes):
  for i in range (0, num_classes):
    if gt_classes_colors[0,i] == gt_class_name:
      break
  return [gt_classes_colors[1,i], i]

######## Helper Code ########

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  np_image = np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype('uint8')
  return np_image

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

    #predictions = np.zeros((len(os.listdir(IMAGE_DIR_PATH)), 1), np.int8)

    for image_path in sorted(os.listdir(IMAGE_DIR_PATH)):
      image_path = os.path.join(IMAGE_DIR_PATH, image_path)
      file_name, ext = os.path.splitext(os.path.basename(image_path))
      gt_class_name = file_name.split("_")[0]
      gt_color, class_id = get_class_color(gt_class_name, gt_classes_colors, num_classes)
      image = Image.open(image_path)

      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      print(file_name)
      if image.mode == 'L':
        w, h = image.size
        ima = Image.new('RGB', (w,h))
        data = zip(image.getdata(), image.getdata(), image.getdata())
        ima.putdata(list(data))
        image = ima
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
          max_boxes_to_draw=None,
          min_score_thresh = score_threshold,
          use_normalized_coordinates=True,
          line_thickness=ln_thickness,
          skip_scores=False)

      img = Image.fromarray(image_np, 'RGB')

      # Determine evaluation method based on crop
      if crop == 'potato':
        accuracy = precision_calculator(gt_color, boxes[1])
        #print(accuracy)
        # Increment total image count
        accuracies[1, class_id] = accuracies[1, class_id] + 1
        # Update sum of accuracies so far 
        accuracies[0, class_id] = accuracies[0, class_id] + accuracy
        # Update class average accuracy
        accuracies[2, class_id] = accuracies[0, class_id] / accuracies[1, class_id]
        #print(accuracies)
        
        if accuracy == 1:
          save_path = os.path.join(CORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")
        else:
          save_path = os.path.join(INCORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")
      
      elif crop is 'Cassava':
        result = make_diagnosis(gt_color, class_id, boxes[1])

        if result == 1:
          save_path = os.path.join(CORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")
        elif result == 0:
          save_path = os.path.join(INCORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")
        elif result == 2:
          save_path = os.path.join(TIE_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")

      elif crop is 'banana':
        accuracy = precision_calculator(gt_color, boxes[1])
        if accuracy == 1:
          save_path = os.path.join(CORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")
        else:
          save_path = os.path.join(INCORRECT_OUTPUT_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")

      else:
          save_path = os.path.join(RESULTS_IMAGES_DIR_PATH, (file_name+"_output.jpg"))
          img.save(save_path, "JPEG")



