import xml.etree.cElementTree as ET
import os
import numpy as np
import tensorflow as tf

from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags

flags.DEFINE_string('image_directory', '/home/plantvillage/Desktop/Melissa Axio Zoom Images/2019-03-01/15_p32_rgb_norm', 'Path to the directory of images')
flags.DEFINE_string('model_path', '/home/plantvillage/Dropbox/Object_Detection/spore/model/pb_files/spore_detect_18.0.pb','Path to the model to load')
flags.DEFINE_string('label_map_path', '/home/plantvillage/Dropbox/Object_Detection/spore/data/spore_label_map.pbtxt', 'Path to the label map text file')
flags.DEFINE_string('label_list_path', '/home/plantvillage/Dropbox/Object_Detection/spore/data/spore_label_list.txt', 'Path to the labels list text file')
flags.DEFINE_string('output_directory', '/home/plantvillage/Desktop/Melissa Axio Zoom Images/2019-03-01/15_p32_annotations', 'Path to the directory to save the xmls')
flags.DEFINE_float('threshold', 0.7, 'The threshold for detection confidences')
flags.DEFINE_string('class_label', 'spore', 'The class label to assign each detected object ' )
FLAGS = flags.FLAGS

######## File Path Preparation ########
PATH_TO_MODEL = FLAGS.model_path
PATH_TO_IMAGES_DIR = FLAGS.image_directory
PATH_TO_LABELS = FLAGS.label_map_path
PATH_TO_LABELS_LIST = FLAGS.label_list_path
PATH_TO_XMLS_IMAGES_DIR = FLAGS.output_directory

with open(PATH_TO_LABELS_LIST) as f:
    gt_classes = f.read().splitlines()

# If path to xmls directory doesn't exist, create the directory
if not os.path.exists(PATH_TO_XMLS_IMAGES_DIR):
  os.makedirs(PATH_TO_XMLS_IMAGES_DIR)

score_threshold = FLAGS.threshold
colors = vis_util.STANDARD_COLORS
#gt_classes = ['???','FAWLeaf','FAWFrass']            f.write('\t<object>\n')
            
NUM_CLASSES=len(gt_classes) - 1

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
  image = np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  return image

def count_boxes(boxes):
  return len(boxes)

def get_class_from_color(color):
  for i in range(0,len(colors)):
    if colors[i] == color:
      return(gt_classes[i])


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

    print('\nBeginning Annotation')
    for image_path in sorted(os.listdir(PATH_TO_IMAGES_DIR)):
      file_name, ext = os.path.splitext(os.path.basename(image_path))
      xml_path = os.path.join(PATH_TO_XMLS_IMAGES_DIR, (file_name + '.xml'))
      #if not os.path.isfile(xml_path):
      image = Image.open(os.path.join(PATH_TO_IMAGES_DIR, image_path))
      
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
          use_normalized_coordinates=True,
          max_boxes_to_draw=None,
          min_score_thresh=score_threshold,
          line_thickness=6)

      image_width, image_height = image.size
     
      full_file_name = os.path.basename(image_path)
      #root = ET.Element("annotation")
      #tree = ET.ElementTree(root)
      #tree.write(xml_path)

      folder_name = 'doesnt_matter'
      image_depth = 3
      class_name = FLAGS.class_label
      file_path = '/this/path/doesnt/matter/' + full_file_name

      print('Creating xml for ' + xml_path)
      f = open(xml_path, 'w+')
      # Write the filename, file path 
      f.write('<annotation>\n\t<folder>%s</folder>\n' % (folder_name))
      f.write('\t<filename>%s</filename>\n' % (full_file_name))
      f.write('\t<path>%s</path>\n' % (file_path))
      f.write('\t<source>\n\t\t<database>Unknown</database>\n\t</source>\n\t<size>\n')
      f.write('\t\t<width>%s</width>\n\t\t<height>%s</height>\n\t\t<depth>%s</depth>\n' % (image_width, image_height, image_depth))
      f.write('\t</size>\n\t<segmented>0</segmented>\n')
    
    
      #i = 1
      drawn_boxes = []
      iou_scores = []

      for box, color in boxes[1]:
        #class_name = get_class_from_color(color)
        #print(class_name)
        ymin, xmin, ymax, xmax = box
        '''
        if len(drawn_boxes) == 0:
          print('Adding first box to drawn boxes list')
          drawn_boxes.append(box)
        '''
        #print(drawn_boxes)
        ymin = int(np.floor(ymin*image_height))
        xmin = int(np.floor(xmin*image_width))
        ymax = int(np.floor(ymax*image_height))
        xmax = int(np.floor(xmax*image_width))

        # loop through all boxes that have been drawn
        '''
        for drawn_box in drawn_boxes:
            ymin_drawn, xmin_drawn, ymax_drawn, xmax_drawn = drawn_box
            ymin_drawn = int(np.floor(ymin_drawn*image_height))
            xmin_drawn = int(np.floor(xmin_drawn*image_width))
            ymax_drawn = int(np.floor(ymax_drawn*image_height))
            xmax_drawn = int(np.floor(xmax_drawn*image_width))
            if xmax_drawn != xmax:  
              x_left = max(xmin, xmin_drawn)
              x_right = min(xmax, xmax_drawn)
              y_top = max(ymin, ymin_drawn)
              y_bottom = min(ymax, ymax_drawn)

              # compute the area of intersection rectangle
              interArea = max(0, x_right - x_left + 1) * max(0, y_bottom - y_top + 1)
              # compute the area of both rectangles
              boxNewArea = (xmax - xmin + 1) * (ymax - ymin + 1)
              boxExistingArea = (xmax_drawn - xmin_drawn + 1) * (ymax_drawn - ymin_drawn + 1)
              # compute the intersection over union by taking the intersection
              # area and dividing it by the sum of prediction + ground-truth
              # areas - the interesection area
              iou = interArea / float(boxExistingArea + boxNewArea - interArea)
              print(iou)
              iou_scores.append(iou)
        #print(all(i < 0.85 for i in iou_scores))
        '''
        i = 0
        if all(i < 0.85 for i in iou_scores):
          #print('Adding box to drawn boxes list')
          #drawn_boxes.append(drawn_box)
          # Now lets add the detected objects
          f.write('\t<object>\n')
          f.write('\t\t<name>%s</name>\n' % (class_name))
          f.write('\t\t<pose>Unspecified</pose>\n\t\t<truncated>0</truncated>\n\t\t<difficult>0</difficult>\n\t\t<bndbox>\n')
          f.write('\t\t\t<xmin>%s</xmin>\n\t\t\t<ymin>%s</ymin>\n\t\t\t<xmax>%s</xmax>\n\t\t\t<ymax>%s</ymax>\n\t\t</bndbox>\n\t</object>\n' % (xmin,ymin,xmax,ymax))
        
        

      # Finish off the file
      f.write('</annotation>')
      f.close()
