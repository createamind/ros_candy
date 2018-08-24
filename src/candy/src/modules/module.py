import tensorflow as tf
import os

class Module(object):
    r"""Abstract module for all modules.

    Attributes:
        _args (dict): Arguments.
        _name (string): The name of the module.
        _saver (tf.train.saver): The saver of variables.
        outputs (tuple): Tuple of outputs of the module.

    """
    def __init__(self, args, name, inputs, is_training=False, reuse=False):
        r"""Constructor.

        Args:
            args (dict): Arguments.
            name (string): The name of the module.
            inputs (tuple): Tuple of inputs.
            reuse (bool): If `reuse=True`, the module reuses existing variables. 
        """
        # Private
        self._args = args
        self._name = name

        # Public 
        self.outputs = self._build_net(inputs, is_training, reuse)

        #Variable saver
        self._saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name))

    def _build_net(self, inputs, is_training, reuse):
        r"""Build the graph. The function should be overwritten by sub-classes.

        Args:
            inputs (tuple): Tuple of inputs.
            reuse (bool): If `reuse=True`, the module reuses existing variables. 
        
        Returns:
            tuple: The outputs of the module.
        """
        raise NotImplementedError

    def optimize(self, loss):
        r"""Create optimization operator, using learning rates from arguments.

        Args:
            loss (op): The operator(tensorflow) of the final loss.

        Returns:
            op: The final operator(tensorflow) of optimization
        """
        self.opt = tf.train.AdamOptimizer(learning_rate=self._args[self._name]['learning_rate'])

        # Clip Gradient
        gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name))
        gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]

        opt_op = self.opt.apply_gradients(gvs)
        return opt_op


    def variable_restore(self, sess):
        r"""restore variables from `./save` directory, using self._saver. It is called with a tensorflow session.

        Args:
            sess (tf.Session): The session.

        """
        model_filename = os.path.join("save", self._name)
        if os.path.isfile(model_filename + '.data-00000-of-00001'):
            self._saver.restore(sess, model_filename)
            return