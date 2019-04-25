import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.patches as mpatches 

flags = tf.app.flags
flags.DEFINE_string('counts_file_path','/home/plantvillage/Dropbox/Object_Detection/spore/data/validation_images/23/23.0_counts_val_0.7/spore_counts_v23.0_0.7%.txt','File path to the counts file')
flags.DEFINE_string('version','23','')
flags.DEFINE_string('threshold','0.7','')

FLAGS = flags.FLAGS

def main(_):
  counts_file_path = FLAGS.counts_file_path
  version = FLAGS.version
  threshold = FLAGS.threshold
  data = np.genfromtxt(counts_file_path, usecols=(0,1,2), skip_header=0, dtype=str)
  #data = np.loadtxt(counts_file_path, delimiter="\t")
  #print(data)

 ###### Plot the graph called data ##########
  file_name = data[:,0]
  gt_count = data[:,1].astype(int)
  pred_count = data[:,2].astype(int)
  diff = (pred_count - gt_count) #/ (pred_count.astype(int) + gt_count.astype(int))
  zipped_counts = zip(gt_count, pred_count, file_name, diff)
  zipped_counts.sort()

  gt_count_sorted = [v for v, x, y, z in zipped_counts]
  pred_count_sorted = [x for v, x, y, z in zipped_counts]
  file_num_sorted = [y for v, x, y, z in zipped_counts]
  diff_sorted = [z for v, x, y, z in zipped_counts]
  
  fig = plt.figure()
  ax1 = plt.subplot(111) 
  ax1.xaxis.grid(which='major', color='k', linestyle='-', linewidth=0.25)

  gt_count_patch = mpatches.Patch(color='red', label='Ground Truth Count')
  pred_count_patch = mpatches.Patch(color='black', label='Predicted Count')
  diff_count_patch = mpatches.Patch(color='blue', label='Count Difference')
  ax1.legend(handles=[gt_count_patch, pred_count_patch, diff_count_patch], loc='best')

  title = 'Spore v' + version + '(' + threshold + ') Validation Counts'
  ax1.set_xlabel('Image Number')
  ax1.set_ylabel('Count') 
  ax1.set_title(title)

  #print(file_name)

  area= np.pi*30
  a = np.arange(len(file_name))

  ax1.scatter(a, gt_count_sorted, c='r', s=area, marker='x')
  ax1.scatter(a, pred_count_sorted, c='k', s=area, marker='.')
 
  ax1.set_xticks(a)
  plt.xticks(rotation=90)
  ax1.set_xticklabels(file_num_sorted)
  plt.ylim((-1,30))
  #print file_num_sorted

  ax2 = ax1.twinx()
  ax2.locator_params(nbins=8, axis='y')
  
  ax2.scatter(a, diff_sorted, c='b', s=area/3, marker='D')
  ax2.set_ylabel('Count Difference')
  plt.ylim((-30,10))
 
  plt.show()

if __name__ == '__main__':
  tf.app.run()

