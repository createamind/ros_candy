import tensorflow as tf
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

    def _encode(self):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])
        
        def conv(x, filters, filter_size, strides=1): 
            return tf.layers.conv2d(x, filters, filter_size, 
                                    strides=strides, padding='same', 
                                    kernel_initializer=utils.kaiming_initializer(), 
                                    kernel_regularizer=l2_regularizer)

        def conv_bn_relu(x, filters, filter_size, strides=1):
            x = conv(x, filters, filter_size, strides)
            x = utils.bn_relu(x, self.is_training)

            return x
        
        x = self.inputs

        # record an original image
        if not self.reuse:
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image('original_image', timage[:1])
        
        # encoder net
        with tf.variable_scope('encoder', reuse=self.reuse):
            # x = 320, 32, 3
            x = conv_bn_relu(x, 16, 7, 2)
            
            # x = 160, 160, 16
            x = conv_bn_relu(x, 32, 5, 2)
            
            # x = 80, 80, 32
            x = conv_bn_relu(x, 64, 5, 2)

            # x = 40, 40, 64
            x = conv_bn_relu(x, 128, 3, 2)

            # x = 20, 20, 128
            x = conv_bn_relu(x, 256, 3, 2)

            # x = 10, 10, 256
            x = conv_bn_relu(x, 512, 3, 2)
            
            # x = 5, 5, 512
            x = tf.reshape(x, [-1, 12800])

            x = tf.layers.dense(x, 512, 
                                kernel_initializer=utils.kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = utils.bn_relu(x, self.is_training)
            x = tf.layers.dense(x, 2 * self.z_size, kernel_initializer=utils.xavier_initializer(), kernel_regularizer=l2_regularizer)

            mean, logstd = tf.split(x, 2, 1)

            # record some weights
        if not self.reuse:
            with tf.variable_scope('encoder', reuse=True):
                w = tf.get_variable('conv2d/kernel')
                tf.summary.histogram('conv0_weights', w)
                w = tf.get_variable('dense_1/kernel')
                tf.summary.histogram('dense1_weights', w)

        return mean, logstd

    def _sample_z(self, mean, logstd):
        with tf.variable_scope('sample_z', reuse=self.reuse):
            std = tf.exp(logstd)
            epsilon = tf.random_normal(tf.shape(mean))

            sample_z = mean + std * epsilon
            tf.summary.histogram('sample_z', sample_z)

        return sample_z

    def _decode(self, sample_z, reuse):
        l2_regularizer = tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay'])
        
        def conv_transpose(x, filters, filter_size, strides=1, kernel_initializer=utils.kaiming_initializer()): 
            return tf.layers.conv2d_transpose(x, filters, filter_size, strides=strides, padding='same', 
                                              kernel_initializer=kernel_initializer, kernel_regularizer=l2_regularizer)
        
        def convtrans_bn_relu(x, filters, filter_size, strides=1):
            x = conv_transpose(x, filters, filter_size, strides)
            x = utils.bn_relu(x, self.is_training)
            return x

        x = sample_z

        # decoder net
        with tf.variable_scope('decoder', reuse=reuse):
            x = tf.layers.dense(x, 512, 
                                kernel_initializer=utils.xavier_initializer(), kernel_regularizer=l2_regularizer)
            x = tf.layers.dense(x, 12800, 
                                kernel_initializer=utils.kaiming_initializer(), kernel_regularizer=l2_regularizer)
            x = utils.bn_relu(x, self.is_training)
            
            x = tf.reshape(x, [-1, 5, 5, 512])
            # x = 5, 5, 512

            x = convtrans_bn_relu(x, 256, 3, 2)
            # x = 10, 10, 256
            
            x = convtrans_bn_relu(x, 128, 3, 2)
            # x = 20, 20, 128
            
            x = convtrans_bn_relu(x, 64, 5, 2)
            # x = 40, 40, 64
            
            x = convtrans_bn_relu(x, 32, 5, 2)
            # x = 80, 80, 32
            
            x = convtrans_bn_relu(x, 16, 5, 2)
            # x = 160, 160, 16
            
            x = conv_transpose(x, 3, 7, 2, utils.xavier_initializer())
            # x = 320, 320, 3
            outputs = tf.nn.tanh(x)
            
        if not self.reuse:
            with tf.variable_scope('decoder', reuse=True):
                # record some weights
                w = tf.get_variable('conv2d_transpose_5/kernel')
                tf.summary.histogram('conv_trans5_weights', w)
                w = tf.get_variable('dense/kernel')
                tf.summary.histogram('dense_weights', w)

            # record a generated image
            timage = tf.cast((tf.clip_by_value(x, -1, 1) + 1) * 127, tf.uint8)
            tf.summary.image('generated_image', timage[:1])

        return outputs

    def _loss(self, mean, logstd, predictions, labels):
        with tf.variable_scope('loss', reuse=self.reuse):
            with tf.variable_scope('kl_loss', reuse=self.reuse):
                KL_loss = utils.kl_loss(mean, logstd)
                MAX_BETA = 1
                beta = tf.get_variable('beta', shape=(), initializer=tf.constant_initializer([0.01]), trainable=False, dtype=tf.float32)
                new_beta = tf.assign(beta, tf.minimum(1.01 * beta, MAX_BETA))
                beta_KL = beta * KL_loss

            with tf.variable_scope('reconstruction_error', reuse=self.reuse):
                reconstruction_loss = utils.mean_square_error(labels, predictions)
            
            l2_loss = tf.losses.get_regularization_loss(self._name, name='l2_loss')
            
            with tf.control_dependencies([new_beta]):
                loss = reconstruction_loss + beta_KL + l2_loss

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
        self.mean, self.logstd = self._encode()
        self.sample_z = self._sample_z(self.mean, self.logstd)
        self.outputs = self._decode(self.sample_z, reuse=self.reuse)
        self.loss = self._loss(self.mean, self.logstd, self.outputs, self.inputs)
        self.opt_op = self._optimize(self.loss)


    def generate(self, prior=True, num_images=1):
        with tf.variable_scope(self._name, reuse=True):
            if prior == True:
                sample_z = tf.random_normal((num_images, self.z_size))
                outputs = self._decode(sample_z, reuse=True)
            else:
                outputs = self.outputs
            return outputs

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
