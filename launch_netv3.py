from __future__ import absolute_import # managing packages from absolute path
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from konsensOp import *
import argparse
import Kinput

def launch_net():
  with tf.Graph().as_default(): 
    global_step = tf.Variable(0,trainable = False)

    input_images = Kinput.data
    input_labels = Kinput.labels  
    images, labels = tf.train.batch_join(
          [input_images, input_labels],batch_size=args.bsize, capacity = 400, enqueue_many = True, dynamic_pad = False) 
    #label = tf.cast(label, tf.int32)
    #images, labels = tf.train.batch([image, label], batch_size=args.bsize) 
    
    logits = inference(images, args, True)
    
    total_loss = loss(logits, labels, args)
    
    train_op = train(total_loss,global_step, args)
    
    saver = tf.train.Saver(tf.all_variables(),max_to_keep=10)
    summary_op = tf.merge_all_summaries()
    init  = tf.group(tf.initialize_all_variables(),
                       tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(args.train_dir, sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
      while not coord.should_stop(): 

        global_step_value = sess.run(global_step)
        _, loss_value = sess.run([logits,total_loss])
        sess.run([train_op])
        if global_step_value % 10 == 0:
          saver.save(sess, args.checkpoint_path, global_step=global_step)
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, global_step_value)
          print(loss_value, global_step_value,'printing global step')
    except tf.errors.OutOfRangeError:
      print('Saving')
      saver.save(sess, args.train_dir, global_step=global_step_value)
      print('Done training for %d epochs, %d steps.' % (args.num_epochs, global_step_value))
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      
    coord.join(threads)
    sess.close()
       
  

def main(argv=None):
#Parse hyperparameters
  parser = argparse.ArgumentParser()
  parser.add_argument('--bsize', type=int, default=300)
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
  parser.add_argument('--num_epochs', type=int, default=100) 
  parser.add_argument('--NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN', type=int, default=7657)
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

