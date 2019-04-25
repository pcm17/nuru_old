import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import matplotlib.patches as mpatches 

flags = tf.app.flags
flags.DEFINE_string('counts_file_path','/home/plantvillage/Dropbox/Object_Detection/SLF/BlackInstar/gt_counts.txt','File path to the counts file')
FLAGS = flags.FLAGS

def main(_):
  counts_file_path = FLAGS.counts_file_path
  data = np.loadtxt(counts_file_path, delimiter="\t")
  print(data)

 ###### Plot the graph called data ##########
  file_num = data[:,0]
  gt_count = data[:,1]
  pred_count = data[:,2]
  diff = (pred_count - gt_count)
  zipped_counts = zip(gt_count, pred_count, file_num, diff)
  zipped_counts.sort()
  print('sorted counts')
  print(zipped_counts)

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

  ax1.set_xlabel('Image Number')
  ax1.set_ylabel('Count') 
  ax1.set_title('SLF Nymph Counts')

  print(str(file_num))

  area= np.pi*30
  a = np.arange(len(file_num))

  ax1.scatter(a, gt_count_sorted, c='r', s=area, marker='x')
  ax1.scatter(a, pred_count_sorted, c='k', s=area, marker='.')
 
  ax1.set_xticks(a)
  ax1.set_xticklabels(file_num_sorted)
  print file_num_sorted

  ax2 = ax1.twinx()
  ax2.locator_params(nbins=10, axis='y')
  
  ax2.scatter(a, diff_sorted, c='b', s=area/3, marker='D')
  ax2.set_ylabel('Count Difference')
 
  plt.show()

if __name__ == '__main__':
  tf.app.run()

