import os
import tensorflow as tf
import numpy as np

import Common

DEFAULT_LABEL = [0]
DEFAULT_FEATURE = [0.0]


class CVSReader(object):

    def __init__(self, filenamequeue, node_count, skiplinenum=0, labelpos=0):
        self.__filename = filenamequeue
        self.__nodecount = node_count
        self.__skiplinenum = skiplinenum
        self.__labelpos = labelpos

    # 读取函数定义
    def data_transform(self, a, ):
        print (a)

        b = np.zeros([28, 28])
        for i in range(0, 27):
            for j in range(0, 27):
                b[i][j] = a[28 * i + j]
        return b

    def read_data(self, file_queue):
        # skip over __skiplinenum of lines
        reader = tf.TextLineReader(skip_header_lines=self.__skiplinenum)
        key, value = reader.read(file_queue)

        defaults = [DEFAULT_FEATURE for i in range(0, self.__nodecount)]
        defaults.insert(self.__labelpos, DEFAULT_LABEL)

        train_item = tf.decode_csv(value, defaults)

        # normallize datums
        feature = tf.multiply(1.0 / 255.0, train_item[1:])
        label = train_item[0:1]

        return feature, label

    def create_pipeline(self, batch_size=1, num_epochs=None):
        file_queue = tf.train.string_input_producer(self.__filename, shuffle=True, num_epochs=num_epochs)
        example, label = self.read_data(file_queue)
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size

        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, num_threads=1
        )
        return example_batch, label_batch


def Test(filename=[], feature_num=0, batch_size=100):
    reader = CVSReader(filename, feature_num, skiplinenum=1, labelpos=0)
    xs, ys = reader.create_pipeline(batch_size=batch_size, num_epochs=1000)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:

            count = 0
            while not coord.should_stop():
                xs_batch, ys_batch = sess.run([xs, ys])

                print (xs_batch, Common.oneShot(ys_batch, 10))


            count = count + 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    coord.join(threads)


if __name__ == '__main__':
    BATCH_SIZE = 100
    FEATURE_NUM = 784
    FILENAME = "../MNIST_CSV/train.csv"
    Test(filename=[FILENAME], feature_num=FEATURE_NUM, batch_size=BATCH_SIZE)
