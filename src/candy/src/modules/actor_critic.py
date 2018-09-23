import tensorflow as tf
from modules.module import Module

"""
 These classes are designed to cooperate with DDPG,
 which means they are not supposed to be used independently
"""
class Actor(Module):
    def __init__(self, name, args, env_info, action_size, reuse=False, build_graph=True, log_tensorboard=True, is_target=False):
        self.action_size = action_size
        self.state = env_info['state']
        self.is_target = is_target
        super(Actor, self).__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    def _build_graph(self):
        self.action = self._network(self.state)

        if not self.is_target:
            # following operations will be further defined in DDPG
            l2_loss = tf.losses.get_regularization_loss(self.name)
            self.loss = l2_loss
            self.opt_op = None
    
    def _network(self, state):
        with tf.variable_scope('actor_net', reuse=self.reuse):
            kaiming_initializer = tf_utils.kaiming_initializer(uniform=True)
            xavier_initializer = tf_utils.xavier_initializer(uniform=True)
            x = tf.nn.relu(self._dense(state, 256, kernel_initializer=kaiming_initializer))
            x = tf.nn.relu(self._dense(x, 128, kernel_initializer=kaiming_initializer))
            x = tf.tanh(self._dense(x, self.action_size, xavier_initializer))

        return x


class Critic(Module):
    def __init__(self, name, args, env_info, action=None, reuse=False, build_graph=True, log_tensorboard=True, is_target=False):
        self.state = env_info['state']
        self.action = env_info['action'] if action is None else action
        self.is_target = is_target
        super(Critic, self).__init__(name, args, reuse=reuse, build_graph=build_graph, log_tensorboard=log_tensorboard)

    def _build_graph(self):
        self.Q = self._network(self.state, self.action)
        
        if not self.is_target:
            # following operations will be further defined in DDPG
            l2_loss = tf.losses.get_regularization_loss(self.name)
            self.loss = l2_loss
            self.opt_op = None

    def _network(self, state, action):
        with tf.variable_scope('critic_net', reuse=self.reuse):
            kaiming_initializer = tf_utils.kaiming_initializer(uniform=True)
            xavier_initializer = tf_utils.xavier_initializer(uniform=True)
            x = tf.nn.relu(tc.layers.layer_norm(self._dense(state, 256, kernel_initializer=kaiming_initializer)))
            x = tf.concat([x, action], 1)
            x = tf.nn.relu(tc.layers.layer_norm(self._dense(x, 128, kernel_initializer=kaiming_initializer)))
            x = self._dense(x, 1, kernel_initializer=xavier_initializer)

        return x


