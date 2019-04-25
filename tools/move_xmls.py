# This script moves a set of xml files based on a folder of images
import tensorflow as tf
import os
import shutil

######## Flag Definitions ########
flags = tf.app.flags
flags.DEFINE_string('image_directory', '/home/plantvillage/Dropbox/Object_Detection/warehouse/Banana/Images/Yellow_Sigatoka', 'Path to the directory of images')
flags.DEFINE_string('input_xml_directory','/home/plantvillage/Dropbox/Object_Detection/warehouse/Banana/Annotations/Yellow_Sigatoka','Path to the directory to look for the xmls')
flags.DEFINE_string('output_xml_directory', '/home/plantvillage/Dropbox/Object_Detection/warehouse/Banana/Annotations/test', 'Path to save the xmls when they are found')
FLAGS = flags.FLAGS

# Read in folder of images
img_dir = FLAGS.image_directory

# Read in folder of xmls to look through
in_xmls = FLAGS.input_xml_directory

# Read in folder to move xmls to 
out_xmls_path = FLAGS.output_xml_directory

# Loop through the images and look for the cooresponding xml in the input_xml_directory
for image_path in sorted(os.listdir(img_dir)):
  # Separate image name from folder path
  image_name = os.path.basename(image_path)
  
  # Separate image name from extension
  name_without_path = os.path.splitext(image_name)[0]
  
  # Concatenate image name with .xml extension
  xml_name = name_without_path + '.xml'
  
  # Concatenate in and out xml paths with the xml name
  in_xml_path = os.path.join(in_xmls, xml_name)

  # Make sure there is an xml for this image
  try:
			open(in_xml_path)
  except IOError:
    print("There is no xml for the image: " + image_name)
    continue

  out_xml_path = os.path.join(out_xmls_path, xml_name)

  # Move the xml to the output_xml_directory 
  os.rename(in_xml_path, out_xml_path)
  #shutil.move(in_xml_path, out_xml_path)
