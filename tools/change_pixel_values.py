from PIL import Image
import tensorflow as tf
import numpy as np
import glob
import os

flags = tf.app.flags
flags.DEFINE_string('img_dir', '/home/plantvillage/Desktop/check_pixels', 'Path to the directory of images')
flags.DEFINE_string('result_dir', '/home/plantvillage/Desktop/check_pixels_results', 'Path to the directory of images')

FLAGS = flags.FLAGS

img_dir = FLAGS.img_dir
result_dir = FLAGS.result_dir

if not os.path.exists(result_dir):
  os.makedirs(result_dir)
  print("\nCreating new output directory")

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
    for col in range(im_width):
        for row in range(im_height):
            pixel = image_np[row, col, 1]
            if pixel == 3:
                image_np[row, col, :] = 2
                print('Found Background Pixel!')

    img = Image.fromarray(image_np, 'RGB')

    save_path = os.path.join(result_dir, (filename))
    img.save(save_path, "PNG")

