import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 1000


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))

        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


# ---------------------
# 作者：NBtingwu
# 来源：CSDN
# 原文：https: // blog.csdn.net / searcher_recommeder / article / details / 79482137
# 版权声明：本文为博主原创文章，转载请附上博文链接！