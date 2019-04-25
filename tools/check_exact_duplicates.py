from PIL import Image
import tensorflow as tf
import numpy as np
import glob

flags = tf.app.flags
flags.DEFINE_string('img_dir', '/home/plantvillage/Dropbox/Object_Detection/warehouse/Cassava/images/cassava_dashboard/cassava_capture/unsorted_images_backup', 'Path to the directory of images')
FLAGS = flags.FLAGS


img_dir = FLAGS.img_dir
i = 0
j = 0
# Read in images using Pillow
for img_path in glob.glob(img_dir + '*'):
    j+=1
    for second_img_path in glob.glob(img_dir + '*'):
        if img_path != second_img_path:
            if i % 100 == 0:
                print('Inside loop: %d\tOutside loop: %d' % (i, j))
            i+=1
            image = Image.open(img_path)
            comp_image = Image.open(second_img_path)

            (comp_img_width, comp_img_height) = comp_image.size
            comp_image_np = np.array(comp_image.getdata()).reshape(
                (comp_img_width, comp_img_height, 3)).astype(np.uint8)

            # Convert image into numpy array
            (im_width, im_height) = image.size
            image_np = np.array(image.getdata()).reshape(
                (im_height, im_width, 3)).astype(np.uint8)

            # Look at all columns and all channels of a single row
            row = 0
            comp_row = comp_image_np[row,:,:]
            og_row = image_np[row, :, :]

            if np.array_equal(og_row, comp_row):
                print('Found Duplicate!!')
                print('First image:' + img_path)
                print('Second image:' + second_img_path)