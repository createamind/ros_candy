from __future__ import print_function, division, absolute_import

import tensorflow as tf
import os
from modules.module import Module

class MLP(Module):

    def __init__(self, inputs, size, *args, **kwargs):
        self._size = size
        self._inputs = inputs
        super(MLP, self).__init__(*args, **kwargs)

    def _build_net(self, is_training, reuse):
        x = self._inputs
        
        for i, size in enumerate(self._size):
            x = tf.layers.dense(inputs=x, units=size, activation=None, 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))
                
            if i != len(self._size) - 1:
                x = tf.nn.relu(x)
                x = tf.layers.dropout(inputs=x, rate=0.2, training=is_training)

        return x