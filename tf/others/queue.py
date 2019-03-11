import tensorflow as tf
import os

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
if __name__=="__main__":
    # test01()
    test02()
