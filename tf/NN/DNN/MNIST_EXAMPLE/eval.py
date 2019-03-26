import time
import tensorflow as tf
import inference
import train
import numpy as np

BATCH_SIZE = 600
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./ml/Race/model"
MODEL_NAME = "model.ckpt"

DEFAULT_LABEL = 0
DEFAULT_FEATURE = 0.0

FILENAME = "./Data/eval.csv"

import numpy as np


# from tensorflow.examples.tutorials.mnist import input_data


def oneShot(curr_y_train_batch):
    num_labels = curr_y_train_batch.shape[0]
    index_offset = np.arange(num_labels) * 10
    num_labels_hot = np.zeros((num_labels, 10))
    num_labels_hot.flat[index_offset + curr_y_train_batch.ravel()] = 1.0
    return num_labels_hot


def evalate():
    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-input')

        sess = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        y = inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        coord = tf.train.Coordinator()

        test_data = np.loadtxt(open(FILENAME), delimiter=",", skiprows=1)

        tmpXs = test_data[:, 1:] * 1.0 / 255.0

        ys = np.matrix(test_data[:, 0:1])
        xs = np.matrix(tmpXs)

        curr_x_train_batch = xs.astype(float)
        curr_y_train_batch = ys.astype(int)
        train_batch_labels = oneShot(curr_y_train_batch)

        validata_feed = {x: curr_x_train_batch,
                         y_: train_batch_labels}
        while True:
            ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                accuracy_score = sess.run(accuracy, feed_dict=validata_feed)
                print("After %s training steps,validation accuracy =%g" % (global_step, accuracy_score))
            else:
                print('No checkpoint file found')
                return
            time.sleep(10)

        sess.close()


def main(argv=None):
    evalate()


if __name__ == '__main__':
    tf.app.run()