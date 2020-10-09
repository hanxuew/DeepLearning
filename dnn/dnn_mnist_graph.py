import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
# 定义计算图（graph）
graph = tf.Graph()
with graph.as_default():
    # 定义placeholder，x为输入，y为对应的标签
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
#     W = tf.Variable(tf.zeros([784, 10]))
#     b = tf.Variable(tf.zeros([10]))
    # 定义drop_out存活率
    keep_prof = tf.placeholder(tf.float32)
    # 定义训练的step
    global_steps = tf.Variable(0, trainable=False)
    # 学习率，初始值为0.5，每1000个step衰减为之前的0.5
    learning_rate = tf.train.exponential_decay(0.5, global_steps, 1000, 0.5, staircase=True)
    # logits = tf.matmul(x, W) + b
    # 定义了dense层
    y1 = tf.layers.dense(
        inputs=x,
        units=512,
        activation=tf.nn.tanh,
        bias_initializer=tf.constant_initializer(0),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    y2 = tf.layers.dense(
        inputs=y1,
        units=128,
        activation=tf.nn.relu,
        bias_initializer=tf.constant_initializer(0),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    y_ = tf.layers.dense(
        inputs=y2,
        units=10,
        activation=tf.nn.softmax,
        bias_initializer=tf.constant_initializer(0),
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
    )
    # 定义损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)+tf.contrib.layers.l2_regularizer(0.5)(W))
    # 梯度下降法，降低loss
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_, y), tf.float32))
    # 定义session，graph需要放在session里执行
with tf.Session(graph=graph) as sess:
    # 定义初始化
    init = tf.global_variables_initializer()
    # 执行初始化
    sess.run(init)
    for i in range(100):
        # 按照batch读取训练数据
        batch = mnist.train.next_batch(128)
        # 将图（graph）中定义的操作执行（run）
        _, learn_rate, loss_, acc = sess.run([train_step, learning_rate, loss, accuracy], {x: batch[0], y: batch[1]})
        if i % 100 == 0:
            print("Step:{:>5}, Train Loss:{:>7.4f}, Train Accuracy:{:>7.2%}".format(i, loss_, acc))
            loss_, acc = sess.run([loss, accuracy], {x: mnist.validation.images, y: mnist.validation.labels})
            print("Step:{:>5}, Valid Loss:{:>7.4f}, Valid Accuracy:{:>7.2%}".format(i, loss_, acc))
    loss_, acc = sess.run([loss, accuracy], {x: mnist.test.images, y: mnist.test.labels})
#     W_ = sess.run(W)
#     b_ = sess.run(b)
    print("Training Finished, Loss:{:>7.4f}, Accuracy:{:>7.2%}".format(loss_, acc))
