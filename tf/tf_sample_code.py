import tensorflow  as  tf

print(tf.__version__)

# tf.Variable生成的变量，每次迭代都会变化，
# 这个变量也就是我们要去计算的结果，所以说你要计算什么，你是不是就把什么定义为Variable
'''
TensorFlow程序可以通过tf.device函数来指定运行每一个操作的设备。 

这个设备可以是本地的CPU或者GPU，也可以是某一台远程的服务器。 
TensorFlow会给每一个可用的设备一个名称，tf.device函数可以通过设备的名称，来指定执行运算的设备。比如CPU在TensorFlow中的名称为/cpu:0。 

在默认情况下，即使机器有多个CPU，TensorFlow也不会区分它们，所有的CPU都使用/cpu:0作为名称。 

–而一台机器上不同GPU的名称是不同的，第n个GPU在TensorFlow中的名称为/gpu:n。 
–比如第一个GPU的名称为/gpu:0，第二个GPU名称为/gpu:1，以此类推。 
–TensorFlow提供了一个快捷的方式，来查看运行每一个运算的设备。 
–在生成会话时，可以通过设置log_device_placement参数来打印运行每一个运算的设备。 

–除了可以看到最后的计算结果之外，还可以看到类似“add: /job:localhost/replica:0/task:0/cpu:0”这样的输出 
–这些输出显示了执行每一个运算的设备。比如加法操作add是通过CPU来运行的，因为它的设备名称中包含了/cpu:0。 
–在配置好GPU环境的TensorFlow中，如果操作没有明确地指定运行设备，那么TensorFlow会优先选择GPU
'''
with tf.device('/cpu:0'):
    x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x * x * y + y + 2

# 创建一个计算图的一个上下文环境
# 配置里面是把具体运行过程在哪里执行给打印出来
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 碰到session.run()就会立刻去调用计算
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()



x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# 在with块内部，session被设置为默认的session
with tf.Session() as sess:
    x.initializer.run()     # 等价于 tf.get_default_session().run(x.initializer)
    y.initializer.run()
    result = f.eval()       # 等价于 tf.get_default_session().run(f)
    print(result)


x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# 可以不分别对每个变量去进行初始化
# 并不立即初始化，在run运行的时候才初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

print(result)


x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

init = tf.global_variables_initializer()

# InteractiveSession和常规的Session不同在于，自动默认设置它自己为默认的session
# 即无需放在with块中了，但是这样需要自己来close session
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

# TensorFlow程序会典型的分为两部分，第一部分是创建计算图，叫做构建阶段，
# 这一阶段通常建立表示机器学习模型的的计算图，和需要去训练模型的计算图，
# 第二部分是执行阶段，执行阶段通常运行Loop循环重复训练步骤，每一步训练小批量数据，
# 逐渐的改进模型参数


# 任何创建的节点会自动加入到默认的图
x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

# 大多数情况下上面运行的很好，有时候或许想要管理多个独立的图
# 可以创建一个新的图并且临时使用with块是的它成为默认的图
graph = tf.Graph()
x3 = tf.Variable(3)
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)
print(x2.graph is tf.get_default_graph())

print(x3.graph is tf.get_default_graph())



# 当去计算一个节点的时候，TensorFlow自动计算它依赖的一组节点，并且首先计算依赖的节点
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    # 这里为了去计算z，又重新计算了x和w，除了Variable值，tf是不会缓存其他比如contant等的值的
    # 一个Variable的生命周期是当它的initializer运行的时候开始，到会话session close的时候结束
    print(z.eval())

# 如果我们想要有效的计算y和z，并且又不重复计算w和x两次，我们必须要求TensorFlow计算y和z在一个图里
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)