import argparse
import tensorflow as tf
import os, sys

flags = tf.app.flags
flags.DEFINE_string('image_dir','/home/plantvillage/Dropbox/Object_Detection/banana/data/images','Path to the directory of images')
flags.DEFINE_string('xml_dir', '/home/plantvillage/Dropbox/Object_Detection/banana/data/annotations/xmls', 'Path to the directory of xmls')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/banana/data/images_w_out_annotations', 'Path to the directory to move images')
FLAGS = flags.FLAGS

image_dir = FLAGS.image_dir
xml_dir = FLAGS.xml_dir
output_dir = FLAGS.output_dir

# Try to read in directory of images
if not os.path.exists(image_dir):
  print("\n\nThere is no image directory with the following path:")
  print(image_dir)
  sys.exit()

# Try to read in directory of annotations
if not os.path.exists(xml_dir):
  print("\n\nThere is no xml directory with the following path:")
  print(xml_dir)
  sys.exit()

# If output directory does not already exist, create it
if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# Loop through images
for image_path in os.listdir(image_dir):
  full_image_path = os.path.join(image_dir, image_path)
  #print(image_path)
  file_name, ext = os.path.splitext(os.path.basename(full_image_path))
  xml_name = os.path.join(xml_dir, (file_name + '.xml'))
  #print(xml_name)
  # For each image, look for the annotation in the xml directory
  try:
	  open(xml_name)
  except IOError:
    # If you dont find an xml, move images to new directory
    print("\nThere is no xml for this image. Moving image into output directory")
    os.rename(full_image_path, os.path.join(output_dir, image_path))