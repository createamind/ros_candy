import tensorflow as tf
import os
import sys
import yaml
from modules.utils.utils import save_args

class Module(object):
    def __init__(self, args, name, is_training=False, reuse=False):
        self._args = args
        self._name = name
        self._reuse = reuse
        with tf.variable_scope(self._name, reuse=self._reuse):
            self.outputs = self._build_net(is_training)

        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        if len(collection) > 0:
            self._saver = tf.train.Saver(collection)
        else:
            self._saver = None

    def _build_net(self, is_training):
        raise NotImplementedError

class ModalOps(Module):
    def __init__(self, args, name, is_training=False, reuse=False):
        super(ModalOps, self).__init__(args, name, is_training, reuse)

    def optimize(self, loss):
        learning_rate = self._args[self._name]['learning_rate'] if 'learning_rate' in self._args[self._name] else 1e-3
        beta1 = self._args[self._name]['beta1'] if 'beta1' in self._args[self._name] else 0.9
        beta2 = self._args[self._name]['beta2'] if 'beta2' in self._args[self._name] else 0.999
        grad_clip = self._args[self._name]['grad_clip'] if 'grad_clip' in self._args[self._name] else 5
        
        with tf.variable_scope(self._name, reuse=self._reuse):
            with tf.variable_scope('optimizer', reuse=self._reuse):
                self._optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grad_var_pairs = self._optimizer.compute_gradients(loss, var_list=tf.trainable_variables(self._name))
                    grad_var_pairs = [(tf.clip_by_norm(grad, grad_clip), var) for grad, var in grad_var_pairs]

                    opt_op = self._optimizer.apply_gradients(grad_var_pairs)

        return opt_op

    def variable_restore(self, sess):
        if self._saver is not None:
            key = self._name + '_path_prefix'
            no_such_file = 'Missing_file'
            path_prefix = self._args[key] if key in self._args else no_such_file
            if path_prefix != no_such_file:
                self._saver.restore(sess, path_prefix)
                print("Params for {} are restored".format(self._name))
                return
    
    def save(self, sess):
        if self._saver:
            path_prefix = self._saver.save(sess, os.path.join(sys.path[0], 'saveimage/', str(self._name)))
            key = self._name + '_path_prefix'
            self._args[key] = path_prefix
            save_args({key: path_prefix}, self._args)
            print("Save Complete")