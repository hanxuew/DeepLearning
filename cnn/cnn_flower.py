import cv2
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

w=100
h=100
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

num_example = data.shape[0]
arr = np.arange(num_example)
np.random.shuffle(arr)
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
s = np.int(num_example*ratio)

x_train=data[:s]
y_train=label_oh[:s]
x_vali=data[s:]
y_vali=label_oh[s:]

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

graph = tf.Graph()
with graph.as_default():
    X = tf.placeholder(tf.float32, shape=[None, w, h, c], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 5], name='Y')
    # X = tf.reshape(X, [-1, w * h * c])
    # Y = tf.reshape(Y, [-1, w * h * c])
    keep_prob = tf.placeholder(tf.float32)
    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.5, global_steps, 1000, 0.5, staircase=True)
    # with graph.as_default():
    with tf.name_scope('conv_layers'):
        # re1 = tf.reshape(X, [-1, w * h * c])
        conv1 = tf.layers.conv2d(
            inputs = X,
            filters = 32,
            kernel_size = [5,5],
            padding = 'SAME',
            strides=1,
            activation = tf.nn.relu,
            kernel_initializer = tf.truncated_normal_initializer(mean=.0, stddev=0.01),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(0.03)
        )
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size = [2, 2],
            strides=2
        )
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            strides=1,
            padding="SAME",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
        )
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2 ,2],
            strides=2
        )
    with tf.name_scope('dense_layes'):
        rel = tf.reshape(pool2, [-1, 25 * 25 *64])
        # rel = tf.layers.flatten(pool2)
        densel = tf.layers.dense(
            inputs=rel,
            units=32,
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
        )
        dense_drop = tf.nn.dropout(densel, keep_prob)
        y_ = tf.layers.dense(
            inputs=dense_drop,
            units=5,
            activation=None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003)
        )

epoch = 200
batch_size = 64
with tf.Session(graph=graph) as sess:
    with tf.name_scope("cal_loss"):
        # loss = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_))
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    with tf.name_scope("cal_accuracy"):
        true = tf.argmax(Y, axis=1)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_, 1), tf.int64), true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(epoch):
        for x, y in get_batch(x_train, y_train, batch_size):
        # train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
        # valid_accuracy = accuracy.eval(feed_dict={X: x_vali, Y: y_vali, keep_prob: 1.0})
            l, train_acc, _ = sess.run([loss, accuracy, train_step], feed_dict={X: x,Y: y, keep_prob:1.})
            print('Step %d, loss %g, accuracy %g' % (i, l, train_acc))
    # train_step.run(feed_dict={X: x_vali, Y: y_vali, keep_prob: 1.0})
    # print(' loss %g, accuracy %g' % (i, l, train_acc))
    valid_loss, valid_acc = sess.run([loss, accuracy], feed_dict={X: x_vali, Y: y_vali, keep_prob:1.})



