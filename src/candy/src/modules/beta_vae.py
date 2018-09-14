import tensorflow as tf
import numpy as np
import os
import sys
import yaml
import modules.utils.utils as utils

class BetaVAE(object):
    def __init__(self, name, args, reuse=False):
        self._args = args
        self._name = name
        self.reuse = reuse
        self.z_size = 64
        self.image_size = self._args['image_size']
        self.epsilon = 1e-8
        
        # params for optimizer
        learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999
        
        with tf.variable_scope(self._name, reuse=self.reuse):
            with tf.variable_scope('placeholder', reuse=self.reuse):
                self.inputs = tf.placeholder(tf.float32, (None, self.image_size, self.image_size, 3), name='inputs')
                self.is_training = tf.placeholder(tf.bool, (None), name='is_training')
            
            self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

            collection = tf.global_variables(self._name)
            if len(collection) > 0:
                self._saver = tf.train.Saver(collection)
            else:
                self._saver = None
            
            self._build_graph()
        
            # add image summaries at training time
            with tf.variable_scope('image', reuse=self.reuse):
                # record an original image
                timage = tf.cast((tf.clip_by_value(self.inputs, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('original_image', timage[:1])
                # record a generated image
                timage = tf.cast((tf.clip_by_value(self.x_mean, -1, 1) + 1) * 127, tf.uint8)
                tf.summary.image('generated_image', timage[:1])

    def _encode(self):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])
        
        def conv(x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.kaiming_initializer()): 
            return tf.layers.conv2d(x, filters, filter_size, 
                                    strides=strides, padding=padding, 
                                    kernel_initializer=kernel_initializer, 
                                    kernel_regularizer=l2_regularizer)

        def conv_bn_relu(x, filters, filter_size, strides=1):
            x = conv(x, filters, filter_size, strides)
            x = utils.bn_relu(x, self.is_training)

            return x
        
        x = self.inputs
        
        # encoder net
        with tf.variable_scope('encoder', reuse=self.reuse):
            x = conv_bn_relu(x, 32, 4, 2)           # x = 160, 160, 32
            x = conv_bn_relu(x, 64, 4, 2)           # x = 80, 80, 64
            x = conv_bn_relu(x, 128, 4, 2)          # x = 40, 40, 128
            x = conv_bn_relu(x, 256, 4, 2)          # x = 20, 20, 256
            x = conv_bn_relu(x, 512, 4, 2)          # x = 10, 10, 512
            x = conv_bn_relu(x, 512, 4, 2)          # x = 5, 5, 512
            x = conv(x, 2 * self.z_size, 4, padding='valid', kernel_initializer=utils.xavier_initializer())  
            # x = 1, 1, 2 * z_size

            x = tf.reshape(x, [-1, 2 * self.z_size])
            mean, logsigma = tf.split(x, 2, -1)

            # record some weights
        if not self.reuse:
            with tf.variable_scope('encoder', reuse=True):
                w = tf.get_variable('conv2d/kernel')
                tf.summary.histogram('conv0_weights', w)

        return mean, logsigma

    def _sample_norm(self, mean, logsigma, scope, reuse=None):
        with tf.variable_scope(scope, reuse=self.reuse if reuse is None else reuse):
            std = tf.exp(logsigma)
            epsilon = tf.random_normal(tf.shape(mean))

            sample_z = mean + std * epsilon
            tf.summary.histogram(scope, sample_z)

        return sample_z

    def _decode(self, sample_z, reuse):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])
        
        def conv_transpose(x, filters, filter_size, strides=1, padding='same', kernel_initializer=utils.kaiming_initializer()): 
            return tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding=padding, 
                                              kernel_initializer=kernel_initializer, kernel_regularizer=l2_regularizer)
        
        def convtrans_bn_relu(x, filters, filter_size, strides=1, padding='same'):
            x = conv_transpose(x, filters, filter_size, strides, padding=padding)
            x = utils.bn_relu(x, self.is_training)
            return x

        x = sample_z

        # decoder net
        with tf.variable_scope('decoder', reuse=reuse):
            x = tf.reshape(x, [-1, 1, 1, self.z_size])          # x = 1, 1, z_size

            x = convtrans_bn_relu(x, 512, 5, 1, padding='valid')# x = 5, 5, 512
            x = convtrans_bn_relu(x, 512, 4, 2)                 # x = 10, 10, 512
            x = convtrans_bn_relu(x, 256, 4, 2)                 # x = 20, 20, 256
            x = convtrans_bn_relu(x, 128, 4, 2)                 # x = 40, 40, 128
            x = convtrans_bn_relu(x, 64, 4, 2)                  # x = 80, 80, 64
            x = convtrans_bn_relu(x, 32, 4, 2)                  # x = 160, 160, 32
            x = conv_transpose(x, 3, 4, 2, kernel_initializer=utils.xavier_initializer())
            # x = 320, 320, 3

            x_mean = x
            
        if not self.reuse:
            with tf.variable_scope('decoder', reuse=True):
                # record some weights
                w = tf.get_variable('conv2d_transpose_5/kernel')
                tf.summary.histogram('conv_trans5_weights', w)

        return x_mean

    def _loss(self, mean, logsigma, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                KL_loss = utils.kl_loss(mean, logsigma)
                MAX_BETA = 1
                beta = tf.get_variable('beta', shape=(), initializer=tf.constant_initializer([0.01]), trainable=False, dtype=tf.float32)
                new_beta = tf.assign(beta, tf.minimum(1.01 * beta, MAX_BETA))
                beta_KL = beta * KL_loss / (160**2)

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.control_dependencies([new_beta]):
                loss = reconstruction_loss + beta_KL# + l2_loss

            tf.summary.scalar('reconstruction_error', reconstruction_loss)
            tf.summary.scalar('beta', beta)
            tf.summary.scalar('KL_loss', KL_loss)
            tf.summary.scalar('beta_KL', beta_KL)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total_loss', loss)

            return loss

    def _optimize(self, loss):
        grad_clip = self._args[self._name]['grad_clip'] if 'grad_clip' in self._args[self._name] else 10
        
        with tf.variable_scope('optimizer', reuse=self.reuse):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            grad_var_pairs = self._optimizer.compute_gradients(self.loss, var_list=tf.trainable_variables(self._name))
            if not self.reuse:
                tf.summary.histogram('gradient', grad_var_pairs[0][0])
            grad_var_pairs = [(tf.clip_by_norm(grad, grad_clip), var) for grad, var in grad_var_pairs]

        with tf.control_dependencies(update_ops):
            opt_op = self._optimizer.apply_gradients(grad_var_pairs)

        return opt_op

    def _build_graph(self):
        self.z_mean, self.z_logsigma = self._encode()
        self.sample_z = self._sample_norm(self.z_mean, self.z_logsigma, 'sample_z')
        self.x_mean = self._decode(self.sample_z, reuse=self.reuse)
        self.loss = self._loss(self.z_mean, self.z_logsigma, self.x_mean, self.inputs)
        self.opt_op = self._optimize(self.loss)

    def generate(self, sess, num_images=1):
        with tf.variable_scope(self._name, reuse=True):
            sample_z = np.random.normal(size=(num_images, self.z_size))
            return sess.run(self.x_mean, feed_dict={self.sample_z: sample_z})

    def variable_restore(self, sess):
        if self._saver:
            key = self._name + '_path_prefix'
            no_such_file = 'Missing_file'

            path_prefix = self._args[key] if key in self._args else no_such_file
            if path_prefix != no_such_file:
                try:
                    self._saver.restore(sess, path_prefix)
                    print("Params for {} are restored".format(self._name))
                except:
                    del self._args[key]
    
    def save(self, sess):
        if self._saver:
            path_prefix = self._saver.save(sess, os.path.join(sys.path[0], 'saveimage/trial/', str(self._name)))
            key = self._name + '_path_prefix'
            self._args[key] = path_prefix
            utils.save_args({key: path_prefix}, self._args)
