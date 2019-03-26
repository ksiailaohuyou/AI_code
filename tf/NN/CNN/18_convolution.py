import numpy  as  np
from sklearn.datasets import load_sample_images

import tensorflow  as  tf

import matplotlib.pyplot  as  plt

# 测试卷积核
# 加载数据
# 输入图片通道，[h,w,channels]
# mini-batch 通常是4d    [mini-batch size, h,w,channels]

datasets = np.array(load_sample_images().images, dtype=np.float32)

# 数据包含一个中国庙宇、一个花
batch_size, height, width, channels = datasets.shape

print(batch_size, height, width, channels)

# 创建两个filter
# 高，宽，通道，卷积核
# 7*7，channels，2
filters_test =np.zeros(shape=(7,7,channels,2),dtype=np.float32)


filters_test[:,3,:,0]=1 #垂直
filters_test[3:,:,1]=1  #水平


X=tf.placeholder(tf.float32,shape=(None,height,width,channels))

convolution=tf.nn.conv2d(X,filter=filters_test,strides=[1,2,2,1],padding='SAME')

with  tf.Session() as sess:
    output=sess.run(convolution,feed_dict={X:datasets})

plt.imshow(load_sample_images().images[0])  #绘制第一个图的第一个特征图
plt.show()
plt.imshow(output[0,:,:,0])  #绘制第一个图的第一个特征图
plt.show()
plt.imshow(output[0,:,:,1])  #绘制第一个图的第二个特征图
plt.show()
plt.imshow(output[1,:,:,0])  #绘制第二个图的第一个特征图
plt.show()
plt.imshow(output[1,:,:,1])  #绘制第二个图的第二个特征图
plt.show()


