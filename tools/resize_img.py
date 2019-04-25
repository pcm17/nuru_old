import PIL.Image, os
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/plantvillage/Desktop/pete', 'Root directory to raw images.')
flags.DEFINE_string('method', 'bilinear', 'Tensorflow resize algorithm')
flags.DEFINE_integer('new_width',500,'New width after resizing.')
flags.DEFINE_integer('new_height',500,'New height after resizing.')
FLAGS = flags.FLAGS

def main(_):
  dir_path = FLAGS.data_dir
  resized_height = FLAGS.new_height
  resized_width = FLAGS.new_width
  resize_method = FLAGS.method

  for file_name in os.listdir(dir_path):
    if not "resized" in file_name:
      print('Image: ' + file_name)

      sess = tf.Session()
      ### RESIZE IMAGE ###
      # Read in original image and decode from jpg
      image_decoded = tf.image.decode_jpeg(tf.read_file(os.path.join(dir_path, file_name)), channels=3)

      ## Resized decoded image
      if resize_method.upper().lower() == 'bilinear':
        resized_image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.BILINEAR)
      elif resize_method.upper().lower() == 'area':
        resized_image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.AREA)
      elif resize_method.upper().lower() == 'bicubic':
        resized_image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.BICUBIC)
      elif resize_method.upper().lower() == 'nearest_neighbor':
        resized_image = tf.image.resize_images(image_decoded, [resized_height, resized_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      new_filename, ext = os.path.splitext(os.path.basename(file_name))

      resized_img_path = dir_path + '/resized_' + str(resized_height) + 'x' + str(resized_width) + '/' + os.path.basename(new_filename) + '.jpg'
      ## Cast image to 8 bit int and re-encode image
      encoded_jpg = tf.image.encode_jpeg(tf.cast(resized_image,tf.uint8))
      
      # Write resized image to file
      fwrite=tf.write_file(resized_img_path, encoded_jpg)
      result=sess.run(fwrite)


if __name__ == '__main__':
  tf.app.run()
