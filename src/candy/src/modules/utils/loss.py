import tensorflow as tf

def kl_loss(mu, logsigma):
    return tf.reduce_mean(-0.5 * tf.reduce_sum(1. + 2. * logsigma - mu**2 - tf.exp(2 * logsigma), axis=1), axis=0)



