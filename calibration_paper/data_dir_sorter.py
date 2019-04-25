""" Splits data set into training and validation sets
Most TensorFlow scripts do not have balanced training and validation sets,
meaning that the distribution of images from different classes are not
equal in the training or validation sets. This script shuffles the image
data set from each class and then adds roughly 80 percent into the training
set and the other 20% into the validation set.
"""
import os
import random
import argparse
import shutil

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    '--data_dir',
    type=str,
    default='',
    help='Path to folders of labeled images'
)

PARSER.add_argument(
    '--train_dir',
    type=str,
    default='',
    help='Path to folder of images for training'
)

PARSER.add_argument(
    '--val_dir',
    type=str,
    default='',
    help='Path to folder of images for validation'
)

FLAGS, UNPARSED = PARSER.parse_known_args()

train_list = []
val_list = []

# Looping through images in each class folder
for dir_path, _, file_names in os.walk(FLAGS.data_dir):
    print('in')
    # Create list of absolute file paths
    abs_paths = []
    for f in file_names:
        if '.DS_Store' not in f: # Adding images to list only
            abs_paths.append(os.path.abspath(os.path.join(dir_path, f)))
            print('adding dir path')
    random.seed(6)
    random.shuffle(abs_paths)
    size = len(abs_paths)
    # Splitting images into 80/20 split for training and validation sets
    train_size = int(0.8 * size)
    train_list += abs_paths[:train_size]
    val_list += abs_paths[train_size:]

# Print statements for spot checking
print(train_list)
print(val_list)
print(len(train_list), len(val_list))

# Moving training and validation files to corresponding folders
for train in train_list:
    shutil.move(train, FLAGS.train_dir)
for val in val_list:
    shutil.move(val, FLAGS.val_dir)