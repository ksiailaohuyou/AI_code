import tensorflow as tf
import os
# 定义cifar的数据等命令行参数
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("cifar_dir", "./data/cifar10/cifar-10-batches-bin/", "文件的目录")
tf.app.flags.DEFINE_string("cifar_tfrecords", "./tmp/cifar.tfrecords", "存进tfrecords的文件")
def  test01():
    # 模拟一下同步先处理数据，然后才能取数据训练
    # tensorflow当中，运行操作有依赖性

    # # 1、首先定义队列
    Q = tf.FIFOQueue(3, tf.float32)
    #
    # # 放入一些数据
    enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
    #
    # 2、定义一些处理数据的螺距，取数据的过程      取数据，+1， 入队列
    #
    out_q = Q.dequeue()
    #
    data = out_q + 1
    #
    en_q = Q.enqueue(data)
    #
    with tf.Session() as sess:
        # 初始化队列
        sess.run(enq_many)

        # 处理数据
        for i in range(100):
            sess.run(en_q)

        # 训练数据
        for i in range(Q.size().eval()):
            print(sess.run(Q.dequeue()))

def  test02():


    # 模拟异步子线程 存入样本， 主线程 读取样本

    # 1、定义一个队列，1000
    Q = tf.FIFOQueue(1000, tf.float32)

    # 2、定义要做的事情 循环 值，+1， 放入队列当中
    var = tf.Variable(0.0)

    # 实现一个自增  tf.assign_add
    data = tf.assign_add(var, tf.constant(1.0))

    en_q = Q.enqueue(data)

    # 3、定义队列管理器op, 指定多少个子线程，子线程该干什么事情
    qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)

    # 初始化变量的OP
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 开启线程管理器
        coord = tf.train.Coordinator()

        # 真正开启子线程
        threads = qr.create_threads(sess, coord=coord, start=True)

        # 主线程，不断读取数据训练
        for i in range(300):
            print(sess.run(Q.dequeue()))

        # 回收你
        coord.request_stop()

        coord.join(threads)
def  csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1、构造文件的队列
    file_queue=tf.train.string_input_producer(filelist)

    # 2、构造csv阅读器读取队列数据（按一行）
    reader=tf.TextLineReader()
    key,value=reader.read(file_queue)

    # 3、对每行内容解码
    # record_defaults:指定每一个样本的每一列的类型，指定默认值[["None"], [4.0]]
    records=[['None'],['None']]

    example, label = tf.decode_csv(value, record_defaults=records)



def picread(filelist):
    """
    读取狗图片并转换成张量
    :param filelist: 文件路径+ 名字的列表
    :return: 每张图片的张量
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    print(value)

    # 3、对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)

    print(image)

    # 5、处理图片的大小（统一大小）
    image_resize = tf.image.resize_images(image, [200, 200])

    print(image_resize)

    # 注意：一定要把样本的形状固定 [200, 200, 3],在批处理的时候要求所有数据形状必须定义
    image_resize.set_shape([200, 200, 3])

    print(image_resize)

    # 6、进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)

    print(image_batch)

    return image_batch

if __name__=="__main__":
    # test01()
    # test02()
    # 1、找到文件，放入列表   路径+名字  ->列表当中
    file_name = os.listdir(FLAGS.cifar_dir)

    filelist = [os.path.join(FLAGS.cifar_dir, file) for file in file_name if file[-3:] == "bin"]

    # print(file_name)
    cf = CifarRead(filelist)

    # image_batch, label_batch = cf.read_and_decode()

    image_batch, label_batch = cf.read_from_tfrecords()

    # 开启会话运行结果
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 存进tfrecords文件
        # print("开始存储")
        #
        # cf.write_ro_tfrecords(image_batch, label_batch)
        #
        # print("结束存储")

        # 打印读取的内容
        print(sess.run([image_batch, label_batch]))

        # 回收子线程
        coord.request_stop()

        coord.join(threads)