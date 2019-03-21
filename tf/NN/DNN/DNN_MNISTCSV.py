import  pandas  as pd
import tensorflow  as   tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
import numpy  as  np

train_data=pd.read_csv('./train.csv')

x_train_data=train_data.iloc[:,1:]
y_train_data=train_data['label']


# 构建图阶段
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None,), )
learning_rate = 0.01