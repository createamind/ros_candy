import numpy as np
import tensorflow as tf

def log_normal(sample, mu, logsigma):
    """Compute a Gaussian distribution density with mean mu and standard deviation exp(logsigma)
    
    Arguments:
        sample: x
        mu: mean
        logsigma: log of the standard deviation
    
    Returns:
        log N(x; mu, sigma**2)
    """

    log2pi = tf.constant([np.log(2 * np.pi)])           # log(2pi)
    inverse_sigma = tf.exp(-logsigma)                   # 1/sigma
    tmp = (sample - mu) * inverse_sigma                 # (x - mu)/sigma
    
    return -0.5 * (tmp**2 + 2 * logsigma + log2pi)      # log(N(mu, sigma))

