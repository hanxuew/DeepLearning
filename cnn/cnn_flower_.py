import cv2
import os
import tensorflow as tf
import numpy as np
from skimage import io,transform
import glob
import os

path='../data/flower_data/'
w=100
h=100
c=3

# def read_img(path):
#     imgs=[]
#     labels=[]
#     cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
#     for idx,i in enumerate(cate):
#         for j in os.listdir(i):
#             im = cv2.imread(i+'/'+j)
#             img = cv2.resize(im, (w, h))
#             #print('reading the images:%s'%(i+'/'+j))
#             imgs.append(img)
#             labels.append(idx)
#     return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
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

data,label=read_img(path)

num_example=data.shape[0] # data.shape是(3029, 100, 100, 3)
arr=np.arange(num_example)# 创建等差数组 0，1，...,3028
np.random.shuffle(arr)# 打乱顺序
data=data[arr]
label=label[arr]
print(label)

def to_one_hot(labels):
    l = len(labels)
    res = np.zeros((l, 5), dtype=np.float32)
    for i in range(l):
        res[i][labels[i]] = 1.
    return res
label_oh = to_one_hot(label)

ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label_oh[:s]
x_val=data[s:]
y_val=label_oh[s:]

def gen_batch(dataset, labelset, batchsize):
    for i in range(dataset.shape[0]//batchsize):
        pos = i * batchsize
        x = dataset[pos:pos + batchsize]
        y = labelset[pos:pos + batchsize]
        yield x,y
    remain = np.shape(dataset)[0] % batchsize
    if remain != 0:
        x = dataset[-remain:]
        y = labelset[-remain:]
        yield x,y

graph = tf.Graph()
with graph.as_default():
    X=tf.placeholder(tf.float32,shape=[None,w,h,c],name='X')
    Y=tf.placeholder(tf.int32,shape=[None,5],name='Y')
    keep_prob=tf.placeholder(tf.float32)

with graph.as_default():
    with tf.name_scope('hidden_layers'):
        conv1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # 第二个卷积层(50->25)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # 第三个卷积层(25->12)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

        # 第四个卷积层(12->6)
        conv4 = tf.layers.conv2d(inputs=pool3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    # 全连接层
    with tf.name_scope('full_conneted_layers'):
        print(pool4.get_shape())
        # re1 = tf.reshape(pool4,[-1, 6 * 6 * 128])
        # re1 = tf.reshape(pool4, [-1, np.prod(pool4.get_shape()[1:])])
        re1 = tf.layers.flatten(pool4)
        dense1 = tf.layers.dense(inputs=re1, units=1024, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        h_fc_drop1 = tf.nn.dropout(dense1, keep_prob)
        dense2 = tf.layers.dense(inputs=h_fc_drop1, units=512, activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        h_fc_drop2 = tf.nn.dropout(dense2, keep_prob)
        y_ = tf.layers.dense(inputs=h_fc_drop2, units=5, activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # ---------------------------网络结束-----

with graph.as_default():
    with tf.name_scope("cal_loss"):
        #loss = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_))
    with tf.name_scope("train"):
        train_op=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    with tf.name_scope("cal_accuracy"):
        true = tf.argmax(Y, axis=1)
        correct_prediction = tf.equal(tf.cast(tf.argmax(y_, 1),tf.int64), true)
        acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    summary_loss = tf.summary.scalar('cost', loss)
    summary_acc = tf.summary.scalar('accuracy', acc)

epoch = 200
batch_size = 64
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tmp/visualization/', sess.graph)
    saver = tf.train.Saver()

    step = 0
    for epc in range(epoch):
        for x, y in gen_batch(x_train, y_train, batch_size):
            l, ac, _, s_m = sess.run([loss, acc, train_op, merged], feed_dict={X: x, Y: y, keep_prob: 0.5})
            print("Step: {:>4}, Loss: {:.4f}, Acc: {:.4%}".format(step, l, ac))
            step += 1
            writer.add_summary(s_m, global_step=epc)
    print("Training finished.")
    l, ac, _ = sess.run([loss, acc, train_op],
                        {X: x_val, Y: y_val, keep_prob: 1})
    print("Testing Loss: {:.4f}, Testing Acc: {:.4%}".format(l, ac))