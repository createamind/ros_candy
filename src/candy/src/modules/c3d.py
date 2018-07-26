import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import numpy as np

	
def conv3d(name, l_input, w, b):
	return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)

def max_pool(name, l_input, k):
	return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def inference_c3d(_X, _dropout, batch_size, _weights, _biases):

	# Convolution Layer
	conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
	conv1 = tf.nn.leaky_relu(conv1, name='relu1')
	conv1 = tf.clip_by_value(conv1, -5, 5)
	pool1 = max_pool('pool1', conv1, k=1)

	# Convolution Layer
	conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
	conv2 = tf.nn.leaky_relu(conv2, name='relu2')
	conv2 = tf.clip_by_value(conv2, -5, 5)
	pool2 = max_pool('pool2', conv2, k=2)

	# Convolution Layer
	conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
	conv3 = tf.nn.leaky_relu(conv3, name='relu3a')
	conv3 = tf.clip_by_value(conv3, -5, 5)

	conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
	conv3 = tf.nn.leaky_relu(conv3, name='relu3b')
	conv3 = tf.clip_by_value(conv3, -5, 5)
	pool3 = max_pool('pool3', conv3, k=2)

	# Convolution Layer
	conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
	conv4 = tf.nn.leaky_relu(conv4, name='relu4a')
	conv4 = tf.clip_by_value(conv4, -5, 5)

	conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
	conv4 = tf.nn.leaky_relu(conv4, name='relu4b')
	conv4 = tf.clip_by_value(conv4, -5, 5)


	pool4 = max_pool('pool4', conv4, k=2)

	# Convolution Layer
	conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
	conv5 = tf.nn.leaky_relu(conv5, name='relu5a')
	conv5 = tf.clip_by_value(conv5, -5, 5)

	conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
	conv5 = tf.nn.leaky_relu(conv5, name='relu5b')
	conv5 = tf.clip_by_value(conv5, -5, 5)

	pool5 = max_pool('pool5', conv5, k=2)

	# Fully connected layer
	pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
	dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
	dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

	dense1 = tf.nn.leaky_relu(dense1, name='fc1') # Relu activation
	dense1 = tf.clip_by_value(dense1, -5, 5)
	dense1 = tf.nn.dropout(dense1, _dropout)

	dense2 = tf.nn.leaky_relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
	dense2 = tf.clip_by_value(dense2, -5, 5)

	dense2 = tf.nn.dropout(dense2, _dropout)

	# Output: class prediction
	out = tf.matmul(dense2, _weights['out']) + _biases['out']

	return out



def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var

def _variable_with_weight_decay(name, shape, wd):
	var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
	if wd is not None:
		weight_decay = tf.nn.l2_loss(var) * wd
		tf.add_to_collection('weightdecay_losses', weight_decay)
	return var



class C3D_Encoder(object):
	def __init__(self, args, name, x, reuse=False):

		self.args = args
		self.name = name
		self.x = x
		self.reuse = reuse
		self.define_variables()


	def inference(self):
		self.output = inference_c3d(self.x, 0.5, self.args['batch_size'], self.weights, self.biases)
		if self.reuse==False:
			self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='var_name'))
		return self.output

	
	def define_variables(self):
		with tf.variable_scope('var_name', reuse=self.reuse) as var_scope:
			self.weights = {
				'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], self.args['c3d_encoder']['weight_decay']),
				'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], self.args['c3d_encoder']['weight_decay']),
				'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], self.args['c3d_encoder']['weight_decay']),
				'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], self.args['c3d_encoder']['weight_decay']),
				'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], self.args['c3d_encoder']['weight_decay']),
				'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], self.args['c3d_encoder']['weight_decay']),
				'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], self.args['c3d_encoder']['weight_decay']),
				'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], self.args['c3d_encoder']['weight_decay']),
				'wd1': _variable_with_weight_decay('wd1', [8192, 4096], self.args['c3d_encoder']['weight_decay']),
				'wd2': _variable_with_weight_decay('wd2', [4096, 4096], self.args['c3d_encoder']['weight_decay']),
				'out': _variable_with_weight_decay('wout', [4096, 300], self.args['c3d_encoder']['weight_decay'])
				}
			self.biases = {
				'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
				'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
				'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
				'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
				'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
				'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
				'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
				'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
				'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
				'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
				'out': _variable_with_weight_decay('bout', [300], 0.000),
			}



	# def optimize(self, loss):
	# 	self.opt = tf.train.AdamOptimizer(learning_rate=self.args['c3d_encoder']['learning_rate'])
	# 	opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='var_name'))
	# 	return opt_op


	def optimize(self, loss):
		self.opt = tf.train.AdamOptimizer(learning_rate=self.args[self.name]['learning_rate'])

		gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='var_name'))
		# gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
		opt_op = self.opt.apply_gradients(gvs)

		# opt_op = self.opt.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='var_name'))
		return opt_op


	def variable_restore(self, sess):
		model_filename = os.path.join("save", self.name)

		# if os.path.isfile(model_filename + '.meta'):
		# 	self.saver = tf.train.import_meta_graph(model_filename + '.meta')
		# 	self.saver.restore(sess, model_filename)
		# 	return

		if os.path.isfile(model_filename + '.data-00000-of-00001'):
			self.saver.restore(sess, model_filename)
			return