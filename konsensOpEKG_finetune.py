""" Builds KonsensOp model

Summary of available functions:

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
 
 # Runs evaluation once
 eval_once(saver, summary_writer, top_k_op, summary_op)
 
 # Help functions to efficiently -- add summary for tensorboard visualization: 
 _activation_summary(x)
 
                                 -- specify that a variable is on cpu:
_variable_on_cpu(name, shape, initializer)

                                 -- create a variable with weight decay:
_variable_with_weight_decay(name, shape, stddev, wd)

                                 -- generate moving average for all losses and associated summaries for
  visualizing the performance of the network:
_add_loss_summaries(total_loss):  


# Optional: implements Batch Normalization
batch_norm(x, n_out, phase_train, scope='batch_norm') """


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import tensorflow as tf
import re
import numpy as np
from batch_norm import *
import argparse
from inputEKG import *


#   Compute inference on the model inputs to make a prediction.
def inference(x_tensor, args, mode):

  phase = tf.constant(mode, dtype = bool)
  
  bnorm0 = batch_norm(x_tensor, 1, phase)   
  with tf.variable_scope('conv1') as scope:
    var = tf.Variable(weights_ft[0][0], dtype=tf.float32, name='weights')
    kernel = _variable_with_weight_decay(var,
                                         stddev=5e-2,
                                         wd=0.0005)
    conv = tf.nn.conv2d(bnorm0, kernel, [1,3,1,1], padding = 'SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    bnorm1 = batch_norm(bias, 64, phase)
    conv1 = tf.nn.relu(bnorm1, name=scope.name)
    _activation_summary(conv1)

  pool1 = tf.nn.max_pool(conv1, 
			ksize=[1,20,1,1],
		        strides=[1,2,1,1],
			padding='SAME',
		        name='pool1')
  dropout1 = tf.nn.dropout(pool1, 0.5)

  with tf.variable_scope('conv2') as scope:
    var = tf.Variable(weights_ft[0][1], dtype=tf.float32, name='weights')
    kernel = _variable_with_weight_decay(var,
                                         stddev=5e-2,
                                         wd=0.0005)
    conv = tf.nn.conv2d(dropout1, kernel, [1,3,1,1], padding='SAME')
    biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv,biases)
    bnorm2 = batch_norm(bias, 64, phase)
    conv2 = tf.nn.relu(bnorm2, name=scope.name)
    _activation_summary(conv2)

  pool2 = tf.nn.max_pool(conv2,
			 ksize=[1,10,1,1], 
			 strides=[1,2,1,1], 
			 padding='SAME', 
			 name='pool2')
  dropout2 = tf.nn.dropout(pool2, 0.5)

  with tf.variable_scope('conv3') as scope:
   
    var = tf.Variable(weights_ft[0][2], dtype=tf.float32, name='weights')
    kernel = _variable_with_weight_decay(var,
                                         stddev=5e-2,
                                         wd=0.0005)
    conv = tf.nn.conv2d(dropout2, kernel, [1,10,1,1], padding='SAME')
    biases = _variable_on_cpu('biases',[64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv,biases)
    bnorm3 = batch_norm(bias, 64, phase)
    conv3 = tf.nn.relu(bnorm3, name=scope.name)
    _activation_summary(conv3)

  pool3 = tf.nn.max_pool(conv3,
			 ksize=[1,9,1,1],
			 strides=[1,9,1,1],
			 padding='SAME',
			 name='pool3')
  dropout3 = tf.nn.dropout(pool3, 0.5)

  with tf.variable_scope('fc4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(dropout3, [args.bsize, -1])
    dim = reshape.get_shape()[1].value
    var = tf.Variable(weights_ft[0][3], dtype=tf.float32, name='weights')
    weights = _variable_with_weight_decay(var,
                                         stddev=5e-2,
                                         wd=0.0005)

    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    fc4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(fc4)

  dropout4 = tf.nn.dropout(fc4, 0.5)

  with tf.variable_scope('softmax_linear') as scope:
    var = tf.Variable(weights_ft[0][4], dtype=tf.float32, name='weights')
    weights = _variable_with_weight_decay(var,
                                         stddev=5e-2,
                                         wd=0.0005)
    biases = _variable_on_cpu('biases', [args.NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.nn.relu(tf.matmul(dropout4, weights) + biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
  
  
#   Compute loss of predictions with respect to the labels
def loss(logits,y_tensor, args):

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_tensor, name='cross_entropy_per_example')	
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
# The total loss is defined as the cross entropy loss plus all of the weight
# decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')  
  
def train(total_loss, global_step, args):
  num_batches_per_epoch = args.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /args.bsize
  decay_steps = int(num_batches_per_epoch * args.NUM_EPOCHS_PER_DECAY)
  
  lr = tf.train.exponential_decay(args.INITIAL_LEARNING_RATE, global_step, decay_steps, args.LEARNING_RATE_DECAY_FACTOR, staircase=False)
  ####lr = _learningRate_decay(args, global_step)


  learnrateSummary = tf.scalar_summary('learning_rate', lr)

  loss_averages_op = _add_loss_summaries(total_loss)
#COMPUTE GRADIENTS
  with tf.control_dependencies([loss_averages_op]): 
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

#APPLY GRADIENTS
  apply_gradient_op = opt.apply_gradients(grads, global_step = global_step) # here we increment global_step

#add histograms for trainable variables and gradients 
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

#track the moving averages of all trainable variables
  variable_averages = tf.train.ExponentialMovingAverage(args.MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')
  return train_op 
  
  
def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      returntrai

    # Start the queue runners.
#    coord = tf.train.Coordinator()
#   try:
#      threads = []
#      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
#        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
#                                         start=True))

      num_iter = int(math.ceil(args.num_examples / args.bsize))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * args.bsize
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
#    except Exception as e:  # pylint: disable=broad-except
#      coord.request_stop(e)

#    coord.request_stop()
#   coord.join(threads, stop_grace_period_secs=10)
  
  
  
  
  
def _add_loss_summaries(total_loss):
  """
  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op
  
  
def _activation_summary(x):
	tensor_name = re.sub('_[0-9]*/','', x.op.name) 
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer = initializer, dtype = dtype)
	return var

def _variable_with_weight_decay(var, stddev, wd):

  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

