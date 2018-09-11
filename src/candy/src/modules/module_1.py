import tensorflow as tf
import os
import sys

class Module(object):
    r"""Abstract module for all modules.

    Attributes:
        _args (dict): Arguments.
        _name (string): The name of the module.
        _saver (tf.train.saver): The saver of variables.
        outputs (tuple): Tuple of outputs of the module.

    """
    def __init__(self, args, name, is_training=False, reuse=False):
        r"""Constructor.

        Args:
            args (dict): Arguments.
            name (string): The name of the module.
            inputs (tuple): Tuple of inputs.
            is_training (bool): If it is training (for dropout and bn ops).   
            reuse (bool): If `reuse=True`, the module reuses existing variables. 
        """
        # Private
        self._args = args
        self._name = name

        # Public 
        with tf.variable_scope(self._name, reuse=reuse) as _:
            self.outputs = self._build_net(is_training, reuse)

        #Variable saver
        collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name)
        if len(collection) > 0:
            self._saver = tf.train.Saver(collection)
        else:
            self._saver = None

    def _build_net(self, is_training, reuse):
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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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
        if self._saver is not None:
            model_filename = os.path.join(sys.path[0], "saveimage/", self._name)
            if os.path.isfile(model_filename + '.data-00000-of-00001'):
                self._saver.restore(sess, model_filename)
                return

    def save(self, sess):
        if self._saver is not None:
            self._saver.save(sess, os.path.join(sys.path[0], 'saveimage/', str(self._name)), global_step=None, write_meta_graph=False, write_state=False)
