from PIL import Image
import tensorflow as tf
import numpy as np
import glob
import os

flags = tf.app.flags
flags.DEFINE_string('img_dir', '/home/plantvillage/Desktop/check_pixels', 'Path to the directory of images')
FLAGS = flags.FLAGS


img_dir = FLAGS.img_dir
i = 0
j = 0
# Read in images using Pillow
for img_path in glob.glob(os.path.join(img_dir,  '*')):
    j+=1
    filename= os.path.basename(img_path)
    print(filename)
    print('Checking image pixel values for image number: %d' % j)
    image = Image.open(img_path)

    # Convert image into numpy array
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

    # Look at all columns and all channels of a single row
    row = 500
    og_row = image_np[row, 400:700, :]

    print(og_row)
    print(im_height)