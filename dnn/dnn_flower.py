import cv2
from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np

w=100
h=100
c=3
path='../flower_data/'

#读取图片
def read_img(path):
  # 逐个打开文件夹，并提取图片
  cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
  imgs = []
  labels = []
  # 用于在for循环中得到计数
  for idx, folder in enumerate(cate):
    # 返回所有匹配的文件路径列表
    for im in glob.glob(folder + '/*.jpg'):
      # print('reading the images:%s'%(im))
      img = io.imread(im)
      img = transform.resize(img, (w, h))
      imgs.append(img)
      labels.append(idx)

  return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)  # 变成矩阵格式

data, label = read_img(path)

#打乱数据集顺序

# data.shape是(3029, 100, 100, 3)
num_example=data.shape[0]
# 创建等差数组 0，1，...,3028
arr=np.arange(num_example)
# 打乱顺序
np.random.shuffle(arr)
data=data[arr]
label=label[arr]
print(label)

#独热向量编码

def to_one_hot(labels):
  l = len(labels)
  res = np.zeros((l, 5), dtype=np.float32)
  for i in range(l):
    res[i][labels[i]] = 1.
  return res

label_oh = to_one_hot(label)

#划分训练集 验证集

ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label_oh[:s]
x_vali=data[s:]
y_vali=label_oh[s:]

#生成batch
def gen_batch(dataset, labelset, batchsize):
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

#构建网络
print("beginning...")
graph = tf.Graph()
with graph.as_default():
  X = tf.placeholder(tf.float32, shape=[None, w, h, c], name='X')
  Y = tf.placeholder(tf.int32, shape=[None, 5], name='Y')  # keep_prob = tf.placeholder(tf.float32)
  keep_prob = tf.placeholder(tf.float32)
  global_steps = tf.Variable(0, trainable=False)
  # learning_rate = 0.5
  learning_rate = tf.train.exponential_decay(0.5, global_steps, 1000, 0.5, staircase=True)
  with graph.as_default():
    with tf.name_scope('hidden_layers'):
      re1 = tf.reshape(X, [-1, w * h * c])
      dense1 = tf.layers.dense(inputs=re1, units=1024, activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      dense1_dpt = tf.nn.dropout(dense1, keep_prob=0.3)
      dense2 = tf.layers.dense(inputs=dense1_dpt, units=1024, activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      dense2_dpt = tf.nn.dropout(dense2, keep_prob=0.3)
      dense3 = tf.layers.dense(inputs=dense2_dpt, units=128, activation=tf.nn.relu,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
      dense3_dpt = tf.nn.dropout(dense3, keep_prob=0.3)
    with tf.name_scope("fully_connected"):
      logits = tf.layers.dense(inputs=dense3_dpt, units=5, activation=None,
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

  with graph.as_default():
    with tf.name_scope("cal_loss"):
      # loss = tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))#tf.contrib.layers.l2_regularizer(0.5)(W)
    with tf.name_scope("train"):
      train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope("cal_accuracy"):
      true = tf.argmax(Y, axis=1)
      correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int64), true)
      acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
      l, ac, _, s_m = sess.run([loss, acc, train_op, merged], feed_dict={X: x_train, Y: y_train})
      if step % 500 == 0:
        print("Step: {:>4}, Train Loss: {:.4f}, Train Acc: {:.4%}".format(step, l, ac))
        _, l, ac = sess.run([train_op, loss, acc], feed_dict={X: x_vali, Y: y_vali})
        print("Step: {:>4}, Valid Loss: {:.4f}, Valid Acc: {:.4%}".format(step, l, ac))
      step += 1
      writer.add_summary(s_m, global_step=epc)
  print("Training finished.")
  l, ac, _ = sess.run([loss, acc, train_op], {X: x_vali, Y: y_vali})
  print("Testing Loss: {:.4f}, Testing Acc: {:.4%}".format(l, ac))