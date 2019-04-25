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

j=0

# Read in images using Pillow
for img_path in glob.glob(os.path.join(img_dir,  '*')):
    j+=1
    filename, ext = os.path.splitext(os.path.basename(img_path))

    print(filename)
    print('Checking image pixel values for image number: %d' % j)
    image = Image.open(img_path)

    # Convert image into numpy array
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

    image_np = image_np[:, :, 1]
    img = Image.fromarray(image_np, 'L')

    save_path = os.path.join(result_dir, (filename+".png"))
    img.save(save_path, "PNG")

    print(og_row)