from PIL import Image
import tensorflow as tf
import numpy as np
import glob
import os
import shutil

flags = tf.app.flags
flags.DEFINE_string('img_dir', '/home/plantvillage/Dropbox/Object_Detection/warehouse/bee/images/MOVI0004_frames/unsorted/', 'Path to the directory of images')
flags.DEFINE_string('output_dir', '/home/plantvillage/Dropbox/Object_Detection/warehouse/bee/images/MOVI0004_frames/duplicates/', 'Path to the directory of images')

FLAGS = flags.FLAGS


img_dir = FLAGS.img_dir
i = 0
j = 0
img_list = os.listdir(img_dir)
img_list.sort()
# Read in images using Pillow
image_list = glob.glob(img_dir + '*')
image_list.sort()
num_images = len(image_list)
num_checks = 2
output_dir = FLAGS.output_dir

if not os.path.exists(output_dir):
      os.makedirs(output_dir)

for m in range (0, num_images):

    print('Checking image %d for a duplicate' % m)
    print('On image %s' % os.path.basename(image_list[m]) )
    for n in range (m+1, m + num_checks + 1):
        if m < num_images - num_checks:
            
            img_path = image_list[m]
            second_img_path = image_list[n]
            file_name = os.path.basename(img_path)
            second_file_name = os.path.basename(second_img_path)

            if img_path != second_img_path:
                i+=1
                try:
	                open(img_path)
	                open(second_img_path)
                except IOError:
                    continue
                image = Image.open(img_path)
                comp_image = Image.open(second_img_path)

                (comp_img_width, comp_img_height) = comp_image.size
                comp_image_np = np.array(comp_image.getdata()).reshape(
                    (comp_img_height, comp_img_width, 3)).astype(np.uint8)

                # Convert image into numpy array
                (im_width, im_height) = image.size
                image_np = np.array(image.getdata()).reshape(
                    (im_height, im_width, 3)).astype(np.uint8)

                # Look at all columns and all channels of a single row
                row = 0
                comp_row = comp_image_np[row,:,1]
                og_row = image_np[row, :, 1]

                if np.array_equal(og_row, comp_row):
                    # Move the file to the output directory
                    #os.rename(img_path, os.path.join(output_dir, file_name))
                    os.rename(second_img_path, os.path.join(output_dir, second_file_name))

                    print('\nFound Duplicate!!')
                    print('First image:' + img_path)
                    print('Second image:' + second_img_path)