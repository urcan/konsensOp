## importing packages 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange
import tensorflow as tf
import numpy as np
import trainer




## load my data and labels tensor
data = np.load('/home/joanna/Desktop/cifar10/konsensop/data.npy')
#labels = np.load('/home/joanna/tensorflow-master/tensorflow/models/image/cifar10/konsensop/labels.npy')
labels = np.zeros((16400,))
labels[10001:16400]=1
labels = labels.astype(int)
data = data.astype(np.float32)


def iterate_batches(data, labels, batch_size, num_epochs):
  N = int(data.shape[0])
  batches_per_epoch = int(N/batch_size)
  for i in range(num_epochs):
    for j in range(batches_per_epoch):
      start, stop = j*batch_size, (j+1)*batch_size
      yield data[start:stop,:,:,:], labels[start:stop]


checkpoint_path='/tmp/konsensop'
train_dir = '/tmp/konsensop_train' 
max_steps = 1000000
log_device_placement = True
batch_size = 300 

def train():
  with tf.Graph().as_default():
    global_step = tf.Variable(0,trainable = False)
    x_tensor = tf.placeholder(tf.float32, shape=(batch_size, 3000,1,1))
    y_tensor = tf.placeholder(tf.int64, shape=(batch_size,))
    logits = trainer.inference(x_tensor)
    total_loss = trainer.loss(logits, y_tensor)
    train_op = trainer.train(total_loss,global_step)
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=10)
    summary_op = tf.merge_all_summaries()
    init  = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    for x,y in iterate_batches(data,labels, 300,2):
      global_step_value = sess.run(global_step)
      _, loss_value = sess.run([logits,total_loss], {x_tensor: x, y_tensor: y})
      sess.run([train_op], {x_tensor: x, y_tensor: y})
      if global_step_value % 10 == 0:
        saver.save(sess, checkpoint_path, global_step=global_step)
        summary_str = sess.run(summary_op, {x_tensor:x, y_tensor: y})
        summary_writer.add_summary(summary_str, global_step_value)
      print(loss_value, global_step_value,'printing global step')

def main(argv=None):
 train()

if __name__=='__main__':
 tf.app.run()

#TENSORBOARD: tensorboard --logdir=/tmp/konsensop_train

