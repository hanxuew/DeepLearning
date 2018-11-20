import tensorflow as tf
import params
import data_utils

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, [None, None, 13])
    y = tf.placeholder(tf.int32, [])
    keep_prob = tf.placeholder(tf.float32,[])

    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(params.hidden_size)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    with tf.name_scope('rnn_layer'):
        mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(params.layer_num)], state_is_tuple=True)
        # init_state = mlstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(mlstm_cell,
                                           inputs=x,
                                           sequence_length=None,
                                           initial_state=None,
                                           dtype=tf.float32,
                                           time_major=False)
        output = state[-1][1]
    with tf.name_scope('calculate_accuracy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    with tf.name_scope('calculate_loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid(labels=y, logits=output))
        # y_ = tf.nn.sigmoid(output)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params.epoch):
            for x, y in data_utils.get_batch(x, y, params.batch_size, params.seq_length):
                l, acc, _ = sess.run([loss, acc, train_step], feed_dict={x: x, y: y})
                print("Step: {:>4}, Loss: {:.4f}, Acc: {:.4%}".format(i, l, acc))


