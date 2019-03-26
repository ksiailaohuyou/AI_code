import os
import tensorflow as tf
import numpy as np

def oneShot(label_batch,hot_num):
    num_labels = label_batch.shape[0]
    index_offset = np.arange(num_labels) * hot_num
    num_labels_hot = np.zeros((num_labels, hot_num))
    num_labels_hot.flat[index_offset+label_batch.ravel()] = 1.0
    return num_labels_hot
