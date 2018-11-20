# import matplotlib.pyplot as plt
# import pylab
# import numpy as np
#
# img = plt.imread("1.jpg")[:, :, 0]  # 这里读取图片 3通道取出1通道
# plt.imshow(img, cmap='gray')  # 显示读取的图片
# pylab.show()
#
# # 这个是设置的滤波，也就是卷积核
# filter = np.array([[ -1,-1, 0],
#                 [ -1, 0, 1],
#                 [  0, 1, 1]])
# print(img.shape)
#
# filter_heigh = filter.shape[0]
# filter_width = filter.shape[1]
# conv_heigh = img.shape[0] - filter.shape[0] + 1
# conv_width = img.shape[1] - filter.shape[1] + 1
#
# conv = np.zeros([conv_heigh, conv_width])
# for i in range(conv_heigh):
#     for j in range(conv_width):
#         conv[i][j] = (img[i:i + filter_heigh, j:j + filter_width]*filter).sum()
# plt.imshow(conv, cmap='gray')  # 显示读取的图片
# pylab.show()
# print(conv.shape)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
# 占位符
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder("float", [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_ = tf.nn.softmax(tf.matmul(x, W) + b)

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
keep_prob = tf.placeholder(tf.float32)
# 构建网络
# 卷积层
conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# 池化层
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
# 经过池化层“裁剪”28*28已经变成14*14
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
# 14*14 --7*7

rel = tf.reshape(pool2, [-1, 7 * 7 * 64])
densel = tf.layers.dense(inputs=rel, units=64, activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
densle_drop = tf.nn.dropout(densel, keep_prob)
logits = tf.layers.dense(inputs=densle_drop, units=10, activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

# 定义loss，以及优化方法还有准确率
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))  # tf.argmax() axis=1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 网络初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(64)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print('Step %d, training accuracy %g' % (i, train_accuracy))
            valid_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0})
            print('Step %d, validation accuracy %g' % (i, valid_accuracy))
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0
    }))
# 训练
