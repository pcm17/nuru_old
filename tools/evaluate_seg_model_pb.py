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
from object_detection.utils import ops as utils_ops



######## Flag Definitions ########
flags = tf.app.flags

flags.DEFINE_float('threshold', 0.5, 'The threshold for detection confidences')
flags.DEFINE_string('record_version', '2', 'Path to the directory of images')
flags.DEFINE_string('model_version','2.0','Path to the model to load')
flags.DEFINE_string('crop', 'spore', 'Path to the label map text file')
FLAGS = flags.FLAGS
######## Flag Variable Preparation ########
crop = FLAGS.crop
model_version = FLAGS.model_version
record_version = FLAGS.record_version
score_threshold = FLAGS.threshold

PATH_TO_MODEL = '/home/plantvillage/Dropbox/Object_Detection/' + crop + '/model/pb_files/' + crop + '_seg_' + model_version + '.pb'
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

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

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
      '''
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
      '''
      output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        max_boxes_to_draw=None,
        min_score_thresh = score_threshold,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=ln_thickness)
      #plt.figure(figsize=(12, 8))
      #plt.imshow(image_np)
      img = Image.fromarray(image_np, 'RGB')
      save_path = os.path.join(RESULTS_IMAGES_DIR_PATH, (file_name+"_output.jpg"))
      img.save(save_path, "JPEG")


      