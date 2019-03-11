import tensorflow  as tf

# 创建一张图包含了一组op和tensor,上下文环境
# op:只要使用tensorflow的API定义的函数都是OP
# tensor：就指代的是数据
def  add():
    node1=tf.constant(3.0)
    node2=tf.constant(3.0)


    sum1=tf.add(node1,node2)


    with tf.Session(config=tf.ConfigProto(log_device_placement=True))  as  sess:


        print(sess.run(sum1))

def  graph():
    g=tf.Graph()
    c = tf.constant(1.0)
    with  g.as_default()  :
        a=tf.constant(1.0)
        assert  c.graph is  g


def  placeholder():


    input1=tf.placeholder(tf.float32)
    input2=tf.placeholder(tf.float32)
    output=tf.add(input1,input2)


    with tf.Session() as sess:
        print(sess.run([output],feed_dict={input1:10.0,input2:11.2}))

def  tens():


    input1=tf.placeholder(tf.float32)
    input2=tf.placeholder(tf.float32)
    output=tf.add(input1,input2)


    with tf.Session() as sess:
        print(sess.run([output],feed_dict={input1:10.0,input2:11.2}))
        print(  input2.name)
        print(  input1.name)


def  sum2():
    a=2.0
    b=tf.constant(3.9)
    c=tf.add(a,b)
    with  tf.Session()  as sess:
        print(sess.run(c))
        # print(a)

def  board():


    a = tf.constant(2.0)

    var = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="variable")

    # print(a, var)

    # 必须做一步显示的初始化op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 必须运行初始化op
        sess.run(init_op)

        # 把程序的图结构写入事件文件, graph:把指定的图写进事件文件当中
        filewriter = tf.summary.FileWriter("./tmp/summary/test/", graph=sess.graph)

        print(sess.run([a, var]))
if __name__=='__main__':
    # add()
    # graph()
    # placeholder()
    # tens()
    # sum2()
    board()