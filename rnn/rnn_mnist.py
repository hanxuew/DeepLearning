import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

input_size = 28      # 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
timestep_size = 28   # 时序持续长度为28，即每做一次预测，需要先输入28行
hidden_size = 256    # 隐含层的数量
layer_num = 1        # LSTM layer 的层数
class_num = 10       # 最后输出分类类别数量，如果是回归预测的话应该是 1

_X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, class_num])
# 在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
keep_prob = tf.placeholder(tf.float32, [])

# **RNN 的输入shape = (batch_size, timestep_size, input_size)
X = tf.reshape(_X, [-1, 28, 28])


def lstm_cell():
    cell = tf.contrib.rnn.LSTMCell(hidden_size)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple = True)

# **用全零来初始化state
init_state = mlstm_cell.zero_state(tf.shape(_X)[0], dtype=tf.float32)

# **调用 dynamic_rnn() 来让我们构建好的网络运行起来
# ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
# ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
# ** state.shape = [layer_num, 2, batch_size, hidden_size], 2指的分别是memory cell 和hidden state
# ** 或者，可以取 h_state = state[-1][1] 作为最后输出
# ** 最后输出维度是 [batch_size, hidden_size]
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
# output = outputs[:, -1, :]#batch_size, 最后一个时序输出,hidden_size
output = state[-1][1]

y_rnn = tf.layers.dense(output, 10,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='rnn_output_layer'
                    )
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_rnn))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_rnn, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = mnist.train.next_batch(32)
        _, acc=sess.run([train_step,accuracy],feed_dict={_X: batch[0], y: batch[1],keep_prob: 0.9})
        #print(out)
        #satt=np.asarray(sat)
        #print(satt)
        if i % 10 == 0:
            print('step %d, training accuracy %g' % (i, acc))
            valid_acc = sess.run(accuracy, feed_dict={_X: mnist.validation.images, y: mnist.validation.labels,keep_prob:1})
            print('step %d, validation accuracy %g' % (i, valid_acc))
    ac=sess.run(accuracy,feed_dict={_X: mnist.test.images, y: mnist.test.labels,keep_prob: 1})
    print('test accuracy %g' %ac)
