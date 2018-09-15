import tensorflow as tf
import yaml
import os
import sys

# kaiming initializer
def kaiming_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tf.contrib.layers.variance_scaling_initializer(factor=2, mode='FAN_IN', uniform=uniform, seed=seed, dtype=dtype)

# xavier initializer
def xavier_initializer(uniform=False, seed=None, dtype=tf.float32):
    return tf.contrib.layers.variance_scaling_initializer(factor=1, mode='FAN_AVG', uniform=uniform, seed=seed, dtype=dtype)

# relu and batch normalization
def bn_relu(layer, training): 
    return tf.nn.relu(tf.layers.batch_normalization(layer, training=training))

# mean square error
def mean_square_error(labels, predictions, scope=None):
    return tf.losses.mean_squared_error(labels, predictions, scope=scope)

def kl_loss(mu, logsigma):
    return tf.reduce_mean(-0.5 * tf.reduce_sum(1. + 2. * logsigma - mu**2 - tf.exp(2 * logsigma), axis=1), axis=0)

def default_path(filename):
    return os.path.join(sys.path[0], filename)

# load arguments from args.yaml
def load_args(filename='args.yaml'):
    with open(default_path(filename), 'r') as f:
        try:
            yaml_f = yaml.load(f)
            return yaml_f
        except yaml.YAMLError as exc:
            print(exc)

# save args to args.yaml
def save_args(args, args_to_update=None, filename='args.yaml'):
    if args_to_update is None:
        args_to_update = load_args(filename)

    with open(default_path(filename), 'w') as f:
        try:
            args_to_update.update(args)
            yaml.dump(args_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)
        
def logsumexp(value, axis=None, keepdims=False):
    if axis is not None:
        max_value = tf.reduce_max(value, axis=axis, keepdims=True)
        value0 = value - max_value    # for numerical stability
        if keepdims is False:
            max_value = tf.squeeze(max_value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value0),
                                                axis=axis, keepdims=keepdims))
    else:
        max_value = tf.reduce_max(value)
        return max_value + tf.log(tf.reduce_sum(tf.exp(value - max_value)))