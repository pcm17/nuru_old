import os, glob
import tensorflow as tf
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string('data_dir','/home/plantvillage/Desktop/Melissa Axio Zoom Images/2019-03-01/15_p32_norm','Root directory for images.')
flags.DEFINE_string('output_dir','/home/plantvillage/Desktop/Melissa Axio Zoom Images/2019-03-01/15_p32_rgb_norm','Output directory for images.')
FLAGS = flags.FLAGS

def main(_):
  output_dir = FLAGS.output_dir
  # If the output directory doesnt exist already, create it
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  data_dir = FLAGS.data_dir
  for file_name in glob.glob(data_dir + "/*"):
    print('Image: ' + file_name)
    im = Image.open(file_name)
    w, h = im.size
    ima = Image.new('RGB', (w,h))
    data = zip(im.getdata(), im.getdata(), im.getdata())
    ima.putdata(list(data))
    ima.save(os.path.join(output_dir, os.path.basename(file_name)), "JPEG")
if __name__ == '__main__':
  tf.app.run()
