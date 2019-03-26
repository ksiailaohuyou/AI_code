import os
import tensorflow as tf
import inference
import Reader
import  Common


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 40000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./ml/Race/model"
MODEL_NAME = "model.ckpt"

DEFAULT_LABEL = [0]
DEFAULT_FEATURE = [0.0]
RECORD_NUM = 42000
FEATURE_NUM = 784

FILENAME = "../MNIST_CSV/train.csv"
# TESTFILENAME = "../MNIST_CSV/test01.csv"

def train(filename=[], feature_num=0, batch_size=100):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [None, inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_function"):
        cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entroy_mean = tf.reduce_mean(cross_entroy)
        loss = cross_entroy_mean + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            RECORD_NUM / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter("./logs/", tf.get_default_graph())

    reader = Reader.CVSReader(filename, feature_num, skiplinenum=1, labelpos=0)
    xs, ys = reader.create_pipeline(batch_size=batch_size, num_epochs=1000)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        merged_summary = tf.summary.merge_all()
        try:
            while not coord.should_stop():
                count = 0
                xs_batch, ys_batch = sess.run([xs, ys])
                if count % 100 == 0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    _, loss_value, step, summary = sess.run([train_op, loss, global_step, merged_summary],
                                                            feed_dict={x: xs_batch, y_: Common.oneShot(ys_batch, 10)},
                                                            options=run_options,
                                                            run_metadata=run_metadata)

                    # writer.add_run_metadata(run_metadata, 'step%03d'% count)
                    print("After %d training steps,loss on training batch is %g." % (step, loss_value))
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


                else:
                    _, loss_value, step, summary = sess.run([train_op, loss, global_step, merged_summary],
                                                            feed_dict={x: xs_batch, y_: Common.oneShot(ys_batch, 10)})
                writer.add_summary(summary, step)

                count = count + 1


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')


        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    coord.join(threads)
    writer.close()


def main(argv=None):
    train(filename=[FILENAME], feature_num=FEATURE_NUM, batch_size=BATCH_SIZE)


if __name__ == '__main__':
    tf.app.run()
