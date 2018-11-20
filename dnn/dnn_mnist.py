import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../mnist_data', one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

keep_prof = tf.placeholder(tf.float32)
global_steps = tf.Variable(0, trainable=False)
# learning_rate = 0.5
learning_rate = tf.train.exponential_decay(0.5, global_steps, 1000, 0.5, staircase=True)

#隐藏层
W1 = tf.Variable(tf.truncated_normal([784, 512], mean=0., stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
y1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
y1_dpt = tf.nn.dropout(y1,keep_prof)

W2 = tf.Variable(tf.truncated_normal([512, 64], mean=0., stddev=0.1))
b2 = tf.Variable(tf.zeros([64]))
y2 = tf.nn.relu(tf.matmul(y1_dpt,W2)+b2)
y2_dpt = tf.nn.dropout(y2, keep_prof)

# W3 = tf.Variable(tf.truncated_normal([128, 64], mean=0., stddev=0.1))
# b3 = tf.Variable(tf.zeros([64]))
# y3 = tf.nn.relu(tf.matmul(y2_dpt,W3)+b3)
#y3_dpt = tf.nn.dropout(y3,keep_prof)
#全连接层
W = tf.Variable(tf.truncated_normal([64,10],mean=0.,stddev=0.1))
b = tf.Variable(tf.zeros([10]))
y_ = tf.matmul(y2_dpt,W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_)+tf.contrib.layers.l2_regularizer(0.5)(W))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y), axis=1),tf.argmax(y_, axis=1)),dtype=tf.float32))

sess = tf.Session()
init = tf.initialize_all_variables()
#init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch = mnist.train.next_batch(128)
    _, learn_rate, loss_, acu = sess.run([train_step,learning_rate, loss, accuracy], {x: batch[0], y: batch[1], keep_prof:0.3})
    if i % 100 == 0:
        print("Step:{:>5}, Learn Rate:{:>7.4f}, Train Loss:{:>7.4f}, Train Accuracy:{:>7.2%}".format(i, learn_rate, loss_, acu))
        loss_, acu = sess.run([loss, accuracy], {x: mnist.validation.images, y: mnist.validation.labels, keep_prof:1.})
        print("Step:{:>5}, Learn Rate:{:>7.4f}, Valid Loss:{:>7.4f}, Valid Accuracy:{:>7.2%}".format(i, learn_rate, loss_, acu))

loss_, acu = sess.run([loss, accuracy], {x: mnist.test.images, y: mnist.test.labels, keep_prof:1.})
print("Training Finished, Loss:{:>7.4f}, Accuracy:{:>7.2%}".format(loss_, acu))
sess.close()