from __future__ import print_function, division, absolute_import
import os
import time
import numpy as np
import os.path as osp
import tensorflow as tf
from collections import deque
import gym
from gym.spaces import Box, Discrete, Tuple
# from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from modules.utils.distributions_1 import make_pdtype
from modules.utils.utils import save_args
import sys
from tensorflow.contrib import rnn

HIDDEN = 15

class LstmPolicy(object):

    def __init__(self, args, name, X, nbatch, nsteps, nlstm=10, reuse=False):
        nenv = nbatch // nsteps
        self._args = args
        self._name = name
        self.pdtype = make_pdtype(Box(low=np.array([-1.0,-1.0], dtype=np.float32), high=np.array([1.0,1.0],dtype=np.float32)))
        # X, processed_x = observation_input(ob_space, nbatch)

        # X = tf.placeholder(tf.float32, [nbatch, HIDDEN])
        # M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nsteps, nlstm*2], name='states') #states
        with tf.variable_scope(self._name, reuse=reuse):
            h = X

            lstm_cell = rnn.BasicLSTMCell(nlstm, state_is_tuple=False)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*1,state_is_tuple=False)

            o, snew = cell(h, S)

            h5 = tf.layers.dense(o, 4, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))
            vf = tf.layers.dense(o, 1, kernel_regularizer=tf.contrib.layers.l2_regularizer(self._args[self._name]['weight_decay']))
            
            # h5 = tf.Print(h5, [h5], summarize=15)
            h5 = tf.clip_by_value(h5, -5, 5)

            self.pd, self.pi = self.pdtype.pdfromlatent(h5)


        v0 = vf[:, 0]
        a0 = self.pd.sample()
        a0 = self.pi 
        # a0 = tf.Print(a0, [a0, self.pi], summarize=10)
        a_z = tf.placeholder(tf.float32, [nbatch, 2], name='action_feeder')

        neglogp0 = self.pd.neglogp(a0)
        neglogpz = self.pd.neglogp(a_z)

        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        # self.X = X
        self.S = S
        self.vf = vf


        self.a0 = a0
        self.a_z = a_z
        self.v0 = v0
        self.snew = snew
        self.neglogp0 = neglogp0
        self.neglogpz = neglogpz
        


class PPO(object):
    def __init__(self, args, name, z, ent_coef, vf_coef, max_grad_norm):
        # sess = tf.get_default_session()
        test_z = tf.expand_dims(z[0], 0)
        self._args = args
        self._name = name
        act_model = LstmPolicy(args, 'ppo', test_z, 1, 1, reuse=False)
        train_model = LstmPolicy(args, 'ppo', z, args['batch_size'], args['batch_size'], reuse=True)

        A = tf.placeholder(tf.float32, [None, 2], name='actions')
        ADV = tf.placeholder(tf.float32, [None], name='advantage')
        R = tf.placeholder(tf.float32, [None], name='rewards')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None], name='oldneglogprob')
        OLDVPRED = tf.placeholder(tf.float32, [None], name='oldvalue')
        CLIPRANGE = 0.2


        self.std_action = tf.placeholder(tf.float32, [None, 2], name='stdactions')
        self.std_mask = tf.placeholder(tf.bool, [None], name='stdmask')

        self.A = A
        self.ADV = ADV
        self.R = R
        self.OLDNEGLOGPAC = OLDNEGLOGPAC
        self.OLDVPRED = OLDVPRED


        neglogpac = train_model.pd.neglogp(A)
        # entropy = tf.reduce_mean(train_model.pd.entropy())
        vpred = train_model.vf


        advantages = self.R - self.OLDVPRED


        mean, var = tf.nn.moments(advantages, axes=[0])

        advantages = (advantages - mean) / (tf.sqrt(var) + 1e-5)

        # advantages = tf.Print(advantages, [advantages], summarize=20)
        # observation, action, log_prob, reward, adv_targ, states = sample
        #     # Reshape to do in a single forward pass for all steps
        # _, action_log_probs, dist_entropy, values = self.actor_critic(None, obs=observation, actions=action, states=states, train=True)
        
        dist_entropy = tf.reduce_mean(train_model.pd.entropy())
        ratio = tf.exp(- neglogpac + self.OLDNEGLOGPAC)
        surr1 = ratio * advantages
        surr2 = tf.clip_by_value(ratio, 1.0 - CLIPRANGE,
                                    1.0 + CLIPRANGE) * advantages
        # print(surr1)			
        action_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        # print(action_loss)			
        value_loss = tf.reduce_mean((self.R - vpred) * (self.R - vpred))

        loss = (value_loss * vf_coef + action_loss - dist_entropy * ent_coef)

        # vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # vf_losses1 = tf.square(vpred - R)
        # vf_losses2 = tf.square(vpredclipped - R)
        # vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        # ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        # pg_losses = -ADV * ratio
        # pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        # approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        # clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        # loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        tmd_steer = train_model.pi[:,1] - self.std_action[:,1]
        tmd_thro = train_model.pi[:,0] - self.std_action[:,0]
        imitation_loss = tf.square(tmd_steer) + 0 * tf.square(tmd_thro)
        # tmd = tf.Print(tmd, [tmd], summarize=1000)
        # self.std_action = tf.Print(self.std_action, [self.std_action], summarize=10)
        # imitation_loss = tf.reduce_mean(tf.square(tmd), 1)

        imitation_loss = tf.reduce_mean(tf.boolean_mask(imitation_loss, self.std_mask))
        imitation_loss = tf.where(tf.is_nan(imitation_loss), tf.zeros_like(imitation_loss), imitation_loss)
        loss = 0 * loss + self._args['imitation_coefficient'] * imitation_loss
   
        # tf.summary.scalar('actionloss', action_loss)
        # tf.summary.scalar('valueloss', value_loss)
        # tf.summary.scalar('entropyloss', dist_entropy)
        # tf.summary.scalar('imitation_loss', imitation_loss)
        # with tf.variable_scope('model'):
        #     params = tf.trainable_variables()
        # grads = tf.gradients(loss, params)
        # if max_grad_norm is not None:
        #     grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # grads = list(zip(grads, params))
        # trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        # _train = trainer.apply_gradients(grads)

        # self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        # def save(save_path):
        #     ps = sess.run(params)
        #     joblib.dump(ps, save_path)

        # def load(load_path):
        #     loaded_params = joblib.load(load_path)
        #     restores = []
        #     for p, loaded_p in zip(params, loaded_params):
        #         restores.append(p.assign(loaded_p))
        #     sess.run(restores)
        #     # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.loss = loss
        self.train_model = train_model
        self.act_model = act_model
        self.initial_state = act_model.initial_state
        self._saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name))

        self.opt_op = self.optimize(self.loss)

    def optimize(self, loss):
        self.opt = tf.train.AdamOptimizer(learning_rate=self._args[self._name]['learning_rate'])
        gvs = self.opt.compute_gradients(loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._name))
        # gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if not grad is None]
        opt_op = self.opt.apply_gradients(gvs)
        return opt_op

    def restore(self, sess):
        if self._saver is not None:
            key = self._name + '_path_prefix'
            no_such_file = 'Missing_file'
            path_prefix = self._args[key] if key in self._args else no_such_file
            if path_prefix != no_such_file:
                try:
                    self._saver.restore(sess, path_prefix)
                    print("Params for {} are restored".format(self._name))
                except:
                    del self._args[key]
                return

    def save(self, sess):
        if self._saver:
            path_prefix = self._saver.save(sess, os.path.join(sys.path[0], 'saveimage/trial/', str(self._name)))
            key = self._name + '_path_prefix'
            self._args[key] = path_prefix
            save_args({key: path_prefix}, self._args)