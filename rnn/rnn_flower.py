from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

w=40
h=40
c=3
path='../data/flower_data/'
def read_img(path):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []

    for idx, folder in enumerate(cate):
        for im in glob.glob(folder + '/*.jpg'):
            img = io.imread(im)
            img = transform.resize(img, (w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)

data, label = read_img(path)

def shuffle(num):
    arr = np.arange(num)
    return np.random.shuffle(arr)

data_size = data.shape[0]
arr = shuffle(data_size)
# data = shuffle(data.shape[0], data)
# label = shuffle(data.shape[0], label)
# arr = shuffle(data.shape[0])
data = data[arr]
label = label[arr]

def to_one_hot(labels):
    l = len(labels)
    res = np.zeros((l, 5), dtype = np.float32)
    for i in range(l):
        res[i][labels[i]] = 1.
    return res
label_oh = to_one_hot(label)

ratio = 0.8
s = np.int(data_size*ratio)

x_train=data[:s]
y_train=label_oh[:s]
x_valid=data[s:]
y_valid=label_oh[s:]

graph = tf.Graph()
# mlstm_cell = lstm_cells(2, 256, 0.9)
with graph.as_default():
    def lstm_cells(layer_num, hidden_size, keep_prob):
        cell = tf.contrib.rnn.LSTMCell(hidden_size)
        cell_dpt = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return tf.contrib.rnn.MultiRNNCell([cell_dpt for _ in range(layer_num)], state_is_tuple=True)


    # mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(256, 0.9) for _ in range(layer_num)], state_is_tuple=True)
    def calculate_accuray(labels, logits):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def create_train_step(labels, logits):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        return tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    def get_batch(dataset, labelset, batchsize):
        for i in range(dataset.shape[0] // batchsize):
            pos = i * batchsize
            x = dataset[pos:pos + batchsize]
            y = labelset[pos:pos + batchsize]
            yield x, y
        remain = np.shape(dataset)[0] % batchsize
        if remain != 0:
            x = dataset[-remain:]
            y = labelset[-remain:]
            yield x, y


    # input_size = 40
    # timestep_size = 40
    hidden_size = 256
    layer_num = 2
    keep_prob = tf.placeholder(tf.float32)

    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(hidden_size)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)

    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y = tf.placeholder(tf.float32, shape=[None, 5], name='y')

    with tf.name_scope('conv_layers'):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=1,
            kernel_size=[5,5],
            padding='SAME',
            strides=1,
            activation=tf.nn.relu,
            kernel_initializer = tf.truncated_normal_initializer(mean=.0, stddev=0.01),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.03)
        )
        # pool1 = tf.layers.max_pooling2d(
        #     inputs=conv1,
        #     pool_size=[2, 2],
        #     strides=2
        # )
    conv_ = tf.reshape(conv1, [-1, 40, 40])
    initial_state = mlstm_cell.zero_state(tf.shape(x)[0], dtype=tf.float32)
    with tf.name_scope('recur_layers'):
        outputs, state = tf.nn.dynamic_rnn(
            mlstm_cell,
            inputs=conv_,
            initial_state=initial_state,
            time_major=False)
        output = outputs[:, -1, :]
    with tf.name_scope('dense_layer'):
        y_ = tf.layers.dense(output, 5,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             bias_initializer=tf.constant_initializer(0.1),
                             name='dense_layer'
                             )
epoch = 1000
batch_size = 64
with tf.Session(graph=graph) as sess:
    train_step = create_train_step(x_train, y_train)
    train_accu = calculate_accuray(x_train, y_train)
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        batch = get_batch(x, y, batch_size)
        _, train_acc = sess.run([train_step, train_accu], feed_dict={x: x_train, y: y_train, keep_prob: keep_prob})
        if i % 10 == 0:
            print('step %d, training accuracy %g' % (i, train_acc))
            valid_acc = sess.run(calculate_accuray(x_valid, y_valid), feed_dict={x: x_valid,y: y_valid, keep_prob:1})
            print('step %d, validation accuracy %g' % (i, valid_acc))
    # test_accu = sess.run(calculate_accuray(),feed_dict={_X: mnist.test.images, y: mnist.test.labels,keep_prob: 1})
    print('Finished')