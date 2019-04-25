import io
import os
import PIL.Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('dir', '', 'Root directory containing the images we would like to check their format.')
FLAGS = flags.FLAGS

def main(_):
  directory= FLAGS.dir
  count=0
  for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
      with tf.gfile.GFile(os.path.join(directory,filename), 'rb') as fid:
        encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = PIL.Image.open(encoded_jpg_io)
      if image.format != 'JPEG':
        print('\nImage: ' + filename)
        print('Format: ' + image.format)
        count=count+1
  if count == 0:
    print('\nAll files in ' + os.path.basename(os.path.dirname(directory)) + ' are JPEG Format\n')


if __name__ == '__main__':
  tf.app.run()
