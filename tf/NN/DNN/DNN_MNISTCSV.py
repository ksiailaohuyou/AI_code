import  pandas  as pd
import tensorflow  as   tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
import numpy  as  np
import  csv


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("is_train", 1, "指定程序是预测还是训练")

# 构建图阶段
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None,), )
learning_rate = 0.01


# 构建神经网络层，我们这里两个隐藏层，基本一样，除了输入inputs到每个神经元的连接不同
# 和神经元个数不同
# 输出层也非常相似，只是激活函数从ReLU变成了Softmax而已

# 自己实现隐藏层


def neuron_layer(X, n_neurons, name, activation=None):
    # 包含所有计算节点对于这一层，name_scope可写可不写
    with  tf.variable_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.square(n_inputs)
        # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
        # truncated normal distribution 比 regular normal distribution的值小
        # 不会出现任何大的权重值，确保慢慢的稳健的训练
        # 使用这种标准方差会让收敛快
        # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
        init =tf.truncated_normal((n_inputs,n_neurons),stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        # 向量表达的使用比一条一条加和要高效
        z=tf.matmul(X,w)+b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z
# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#     hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#     # 进入到softmax之前的结果
#     logits = neuron_layer(hidden2, n_outputs, "outputs")


def  get_weight_variable(shapa,regularizer):
    weights=tf.get_variable('weight',shapa,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights
with tf.name_scope("dnn"):
    # tensorflow使用这个函数帮助我们使用合适的初始化w和b的策略，默认使用ReLU激活函数
    hidden1=fully_connected(X, n_hidden1, scope="hidden1",)
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)



with tf.name_scope("loss"):
    # 定义交叉熵损失函数，并且求个样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits只会给one-hot编码，我们使用的会给0-9分类号
    xentropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
    loss=tf.reduce_mean(xentropy)

with tf.name_scope("train"):
    optimizer=tf.train.ProximalGradientDescentOptimizer(learning_rate)
    training_op=optimizer.minimize(loss)

with tf.name_scope("eval"):
    #获取logits里面最大的那1位和y比较类别好是否相同，返回True或者False一组值
    correct=tf.nn.in_top_k(logits,y,1)
    accurary=tf.reduce_mean(tf.cast(correct,tf.float32))






def   read_traindata(filelist,read_num=100):
    """
        读取CSV文件
        :param filelist: 文件路径+名字的列表
        :return: 读取的内容
        """
    # 1、构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [[0.0]]*785
    records[0]=[1]

    # data_line = tf.decode_csv(value, record_defaults=records)
    data_line_list = tf.decode_csv(value, record_defaults=records,)

    # defaults = [DEFAULT_FEATURE for i in range(0, self.__nodecount)]
    # defaults.insert(self.__labelpos, DEFAULT_LABEL)
    #
    # train_item = tf.decode_csv(value, defaults)


    target=  data_line_list[0]
    # feature=data_line_list[1:]
    feature=tf.multiply(1.0 / 255.0, data_line_list[1:])
    # feature=tf.cast(data_line_list[1:],dtype=tf.float32)

    # 4、想要读取多个数据，就需要批处理
    y_targets, x_features = tf.train.batch([target,feature], batch_size=read_num, num_threads=1, capacity=read_num*3)

    # features=tf.cast(features,tf.float32)
    return y_targets,x_features


def   read_test_traindata(filelist,read_num=100):
    """
        读取CSV文件
        :param filelist: 文件路径+名字的列表
        :return: 读取的内容
        """
    # 1、构造文件的队列
    file_queue = tf.train.string_input_producer(filelist)

    # 读取的时候需要跳过第一行
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records = [[0.0]]*784
    # records[0]=[1]

    # data_line = tf.decode_csv(value, record_defaults=records)
    data_line_list = tf.decode_csv(value, record_defaults=records,)

    # defaults = [DEFAULT_FEATURE for i in range(0, self.__nodecount)]
    # defaults.insert(self.__labelpos, DEFAULT_LABEL)
    #
    # train_item = tf.decode_csv(value, defaults)


    # target=  data_line_list[0]
    # feature=data_line_list[1:]
    feature=tf.multiply(1.0 / 255.0, data_line_list)
    # feature=tf.cast(data_line_list[1:],dtype=tf.float32)

    # 4、想要读取多个数据，就需要批处理
    x_features = tf.train.batch([feature], batch_size=read_num, num_threads=1, capacity=read_num*3)

    # features=tf.cast(features,tf.float32)
    return x_features
def  tarin_model():
    init = tf.global_variables_initializer()
    n_epochs = 40
    batch_size = 50

    #
    train_data=pd.read_csv('./MNIST_CSV/train.csv')
    # 定义一个合并变量de op
    # merged=tf.summary.merge_all()
    # 创建一个saver
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()

        if FLAGS.is_train == 1:
            target, features = read_traindata(['./MNIST_CSV/train.csv'], 100)
            # target, features = read_traindata(['./MNIST_CSV/test01.csv'], 100)
            # 定义一个线程协调器
            coord = tf.train.Coordinator()
            # 开启读文件的线程
            threads = tf.train.start_queue_runners(sess, coord=coord)
            # filewriter = tf.summary.FileWriter('./MNIST_CSV/summary/model', graph=sess.graph)
            for epoch in range(n_epochs):
                for iteration in range( train_data.shape[0]//100):
                    y_target, x_features = sess.run([target, features])
                    # y_target_test, x_features_test = sess.run([target_test, features_test])
                    sess.run(training_op, feed_dict={X: x_features, y: y_target})

                    # 写入每步训练的值
                    # summary = sess.run(merged, feed_dict={X: x_features, y: y_target})
                    # filewriter.add_summary(summary, epoch)


                y_target, x_features = sess.run([target, features])
                loss_value=sess.run(loss, feed_dict={X: x_features, y: y_target})
                acc_train = accurary.eval(feed_dict={X: x_features, y:  y_target })

                # acc_test = accurary.eval(feed_dict={X:x_features_test , y:  y_target_test})
                # print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
                print(epoch, "Train accuracy:", acc_train, 'loss_vlue:',loss_value)
                # 保存模型
                saver.save(sess, './MNIST_CSV/ckpt/fc_model')

        else:
            x_test_features = read_test_traindata(['./MNIST_CSV/test.csv'], 28000)
            # 定义一个线程协调器
            coord = tf.train.Coordinator()
            # 开启读文件的线程
            threads = tf.train.start_queue_runners(sess, coord=coord)
            saver.restore(sess,'./MNIST_CSV/ckpt/fc_model')

            # with   open('./MNIST_CSV/sample_submission.csv','w',newline='')  as  csvfile:

                # writer = csv.writer(csvfile)
                # writer.writerow(["ImageId", "Label"])
            x_test = sess.run(x_test_features)
            y_target_list=tf.argmax(sess.run(logits, feed_dict={X: x_test, }), 1).eval()
            sumbuer = pd.read_csv('./MNIST_CSV/sample_submission.csv')
            sumbuer['ImageId']=range(1,28000+1)
            sumbuer['Label']=y_target_list
            sumbuer.to_csv('./MNIST_CSV/sample_submission.csv', index=0, index_label=False)
                # for i in range(0,28000):


                    # for each  in  range(1000):
                    # writer.writerow([i + 1,y_target_list[i] ])
            print('完成')
            # 回收子线程
        coord.request_stop()

        coord.join(threads)

if __name__=='__main__':
    tarin_model()
