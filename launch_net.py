""" Runs training and every 100 steps evaluation of the konsensOp """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from six.moves import xrange
import tensorflow as tf
import numpy as np
from konsensOp import *
import argparse

# fake data
data = np.load('/home/joanna/Desktop/cifar10/konsensop/data.npy')
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
      
      

def launch_net():
  with tf.Graph().as_default():
    global_step = tf.Variable(0,trainable = False)
    
    x_tensor = tf.placeholder(tf.float32, shape=(args.bsize, 3000,1,1))
    y_tensor = tf.placeholder(tf.int64, shape=(args.bsize,))
    
    logits = inference(x_tensor, args, True)
    
    total_loss = loss(logits, y_tensor, args)
    
    train_op = train(total_loss,global_step, args)
    
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=10)
    summary_op = tf.merge_all_summaries()
    init  = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(args.train_dir, sess.graph)

    for x,y in iterate_batches(data,labels, 300,2):
      global_step_value = sess.run(global_step)
      _, loss_value = sess.run([logits,total_loss], {x_tensor: x, y_tensor: y})
      sess.run([train_op], {x_tensor: x, y_tensor: y})
      if global_step_value % 10 == 0:
        saver.save(sess, args.checkpoint_path, global_step=global_step)
        summary_str = sess.run(summary_op, {x_tensor:x, y_tensor: y})
        summary_writer.add_summary(summary_str, global_step_value)
        
        
      print(loss_value, global_step_value,'printing global step')
  

def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('bsize', type=int)
  parser.add_argument('--checkpoint_path', type=str, default='res/saver')
  parser.add_argument('--train_dir', type=str, default='res/trainDir')
  parser.add_argument('--eval_dir', type=str, default='res/evalDir')
  parser.add_argument('--log_device_placement', type=bool,default=True)
  parser.add_argument('--max_steps' , type=int, default= 10000)
  parser.add_argument('--MOVING_AVERAGE_DECAY', type=float, default=0.9999)
  parser.add_argument('--NUM_EPOCHS_PER_DECAY', type=int, default=350)
  parser.add_argument('--LEARNING_RATE_DECAY_FACTOR', type=float, default=0.1)
  parser.add_argument('--INITIAL_LEARNING_RATE', type=float, default=0.1)
  parser.add_argument('--NUM_CLASSES', type=int, default=2) 
  parser.add_argument('--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN', type=int, default=1000)
  parser.add_argument('--eval_data', type=str, default='train_eval', choices=['test', 'train_eval'])
  parser.add_argument('--eval_intervals_sec', type=int, default=60*5)
  parser.add_argument('--num_examples', type=int, default=1000)
  parser.add_argument('--run_once', type=bool,  default=False)
  parser.add_argument('--tensorboard_path', type=str, default='tensorboard --logdir=/tmp/konsensop_train', help= 'Command to visualize tensorboard')
  global args
  args = parser.parse_args()
  launch_net()

if __name__=='__main__':
 main()

