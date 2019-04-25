import argparse
import tensorflow as tf
import os, sys

flags = tf.app.flags
flags.DEFINE_string('image_dir','/home/plantvillage/Dropbox/Object_Detection/spore/data/record18_data/27_p38_rgb_norm','Path to the directory of images')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/spore/data/record18_data/27_p38_subsample', 'Path to the directory to move images')
FLAGS = flags.FLAGS

image_dir = FLAGS.image_dir
output_dir = FLAGS.output_dir

# Try to read in directory of images
if not os.path.exists(image_dir):
  print("\n\nThere is no image directory with the following path:")
  print(image_dir)
  sys.exit()


# If output directory does not already exist, create it
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
interval = 4
# Loop through images
i=0
for image_path in sorted(os.listdir(image_dir)):
  full_image_path = os.path.join(image_dir, image_path)
  #print(image_path)
  file_name, ext = os.path.splitext(os.path.basename(full_image_path))
  #print(xml_name)
  # For each image, look for the annotation in the xml directory
  if i % interval == 0:
    print("\n Moving image image %d into output directory" % (i/interval))
    os.rename(full_image_path, os.path.join(output_dir, image_path))
  i+=1