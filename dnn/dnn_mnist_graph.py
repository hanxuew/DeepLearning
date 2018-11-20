import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)
# 定义计算图
graph = tf.Graph()
with graph.as_default():
    # 定义placeholder
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    keep_prof = tf.placeholder(tf.float32)
    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.5, global_steps, 1000, 0.5, staircase=True)
    # logits = tf.matmul(x, W) + b
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
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)+tf.contrib.layers.l2_regularizer(0.5)(W))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_, y), tf.float32))
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(10000):
        batch = mnist.train.next_batch(128)

        _, learn_rate, loss_, acc = sess.run([train_step, learning_rate, loss, accuracy], {x: batch[0], y: batch[1]})
        if i % 100 == 0:
            print("Step:{:>5}, Train Loss:{:>7.4f}, Train Accuracy:{:>7.2%}".format(i, loss_, acc))
            loss_, acc = sess.run([loss, accuracy], {x: mnist.validation.images, y: mnist.validation.labels})
            print("Step:{:>5}, Valid Loss:{:>7.4f}, Valid Accuracy:{:>7.2%}".format(i, loss_, acc))
    loss_, acc = sess.run([loss, accuracy], {x: mnist.test.images, y: mnist.test.labels})
    W_ = sess.run(W)
    b_ = sess.run(b)
    print("Training Finished, Loss:{:>7.4f}, Accuracy:{:>7.2%}".format(loss_, acc))