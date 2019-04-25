import glob
import random
import os
from shutil import copy2

random.seed(42)
xml_dir = '/home/plantvillage/Dropbox/Object_Detection/warehouse/FAW/annotations/FAWLeaf_annotations/FAWLeaf_Alexis_grouped_annotations/'
img_dir = '/home/plantvillage/Dropbox/Object_Detection/warehouse/FAW/images/FAW_Leaf/Alexis_grouped/'
new_img_dir = '/home/plantvillage/Desktop/rand_select/images/'
new_xml_dir = '/home/plantvillage/Desktop/rand_select/annotations/'

xml_paths = glob.glob(xml_dir + "*")
img_paths = glob.glob(img_dir + "*")
random.shuffle(xml_paths)
jpg = '.jpg'
xml = '.xml'
num_images = 100

# if directory doesnt exist, create it
if not os.path.exists(new_img_dir):
    os.makedirs(new_img_dir)
# do this for xml directory too
if not os.path.exists(new_xml_dir):
    os.makedirs(new_xml_dir)


for i in range(num_images):
  file_name, ext = os.path.splitext(os.path.basename(xml_paths[i]))
  print(img_dir + file_name + jpg)
  # copy image to new directory 
  src = img_dir + file_name + jpg
  dest = new_img_dir
  copy2(src, dest)
  # copy xml file to new directory
  src = xml_dir + file_name + xml
  dest = new_xml_dir
  copy2(src, dest)

