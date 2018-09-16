#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

from modules.beta_vae import BetaVAE
from modules.ppo import PPO

import tensorflow as tf
import numpy as np
import os
import datetime
import functools
import msgpack
import msgpack_numpy as m
m.patch()
from std_msgs.msg import String
from candy.srv import Step, Value, UpdateWeights
from tqdm import tqdm
import rospy
from modules.utils.utils import default_path, load_args
import random
import sys
if not (sys.version_info[0] < 3):
    print = functools.partial(print, flush=True)

class Machine(object):
    def __init__(self):

        args = load_args('args.yaml')
        self._args = args

        #Building Graph
        self.camera = BetaVAE('camera', args, False)
        self.left_eye = BetaVAE('left_eye', args, False)
        self.right_eye = BetaVAE('right_eye', args, False)

        self.speed = tf.placeholder(tf.float32, shape=(self._args['batch_size'], 1), name='speed')
        self.test_speed = tf.placeholder(tf.float32, shape=(1, 1), name='test_speed')

        z = tf.concat([self.camera.z_mu, self.left_eye.z_mu, self.right_eye.z_mu], 1)

        z = tf.clip_by_value(z, -5, 5)

        self.ppo = PPO(self._args, 'ppo', z=z, ent_coef=0.00000001, vf_coef=1, max_grad_norm=0.5)

        self.variable_restore_parts = [self.camera, self.left_eye, self.right_eye]
        self.variable_save_optimize_parts = [self.camera, self.left_eye, self.right_eye, self.ppo]

        self.vae_loss = self.camera.loss + self.left_eye.loss + self.right_eye.loss
        total_loss = self.vae_loss + 0 * self.ppo.loss

        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))

        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
    
        self.final_ops = []
        for part in self.variable_save_optimize_parts:
            self.final_ops.append(part.opt_op)
        self.final_ops = tf.group(self.final_ops)

        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True

        
        self.merged = tf.summary.merge_all()
        self.sess = tf.Session(config = config)
        summary_path = '/tmp/iminlogs/trial/' + self._args['model_name']
        self.writer = tf.summary.FileWriter(summary_path, self.sess.graph)

        with tf.Graph().as_default() as g:
            tf.Graph.finalize(g)
        self.sess.run(tf.global_variables_initializer())

        print('Restoring!')

        for part in self.variable_restore_parts:
            part.restore(self.sess)

        print('Get_Params!')
        self.params = []
        for va in tf.trainable_variables():
            self.params.append(va)
        print(len(self.params), 'params in all!')

        print('Model Started!')

    def step(self, obs, state):
        # mask = np.zeros(1)
        td_map = {self.ppo.act_model.S:state}

        camera_x = np.array([obs[0][0]])
        eye_x1 = np.array([obs[0][2]])
        eye_x2 = np.array([obs[0][3]])
        td_map[self.camera.inputs] = camera_x
        td_map[self.left_eye.inputs] = eye_x1
        td_map[self.right_eye.inputs] = eye_x2

        td_map[self.camera.is_training] = False
        td_map[self.left_eye.is_training] = False
        td_map[self.right_eye.is_training] = False

        td_map[self.test_speed] = np.array([[obs[1]]]) # speed
        
        return self.sess.run([self.ppo.act_model.a0, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogp0, self.vae_loss], td_map)

    def value(self, obs, state, action):
        raise NotImplementedError
        # mask = np.zeros(1)
        # if len(np.array(action).shape) == 1:
        #     action = [action]
        # td_map = {self.ppo.act_model.S:state, self.ppo.act_model.a_z: action}
        # td_map[self.test_raw_image] = np.array([obs[0]])
        # td_map[self.is_training] = False
        # td_map[self.test_speed] = np.array([[obs[1]]])
        # # td_map[self.test_steer] = np.array([[obs[2]]])

        # return self.sess.run([self.ppo.act_model.a_z, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogpz, self.test_vae_loss.recon], td_map)
    
    def update_weights(self, mat):
        # for ind, _ in tqdm(enumerate(self.params)):
        #     self.params[ind].load(mat[ind], self.sess)
        for part in self.variable_restore_parts:
            part.restore(self.sess)

        print('Weights Updated!')

    def train(self, inputs, global_step):
        obs, actions, values, neglogpacs, rewards, vaerecons, states, std_actions, manual = inputs

        values = np.squeeze(values, 1)
        neglogpacs = np.squeeze(neglogpacs, 1)
        actions = np.squeeze(actions, 1)

        # print(raw_image.shape)
        # print(speed.shape)

        advs = rewards - values
        advs = (advs - advs.mean()) / (advs.std() + 1e-5)

        td_map = {self.ppo.A:actions, self.ppo.ADV:advs, self.ppo.R:rewards, self.ppo.OLDNEGLOGPAC:neglogpacs, self.ppo.OLDVPRED:values}
        td_map[self.camera.is_training] = True
        td_map[self.left_eye.is_training] = True
        td_map[self.right_eye.is_training] = True

        # mask = np.zeros(self._args['batch_size'])
        td_map[self.ppo.train_model.S] = np.squeeze(states, 1)
        # td_map[self.ppo.train_model.M] = mask

        td_map[self.ppo.std_action] = std_actions
        td_map[self.ppo.std_mask] = manual
        
        camera_x = np.array([ob[0][random.randint(0, 1)] for ob in obs])
        eye_x1 = np.array([ob[0][2] for ob in obs])
        eye_x2 = np.array([ob[0][3] for ob in obs])
        td_map[self.camera.inputs] = camera_x
        td_map[self.left_eye.inputs] = eye_x1
        td_map[self.right_eye.inputs] = eye_x2
        
        td_map[self.speed] = np.array([[ob[1]] for ob in obs])

        td_map[self.test_speed] = np.array([[obs[0][1]]]) # speed

        summary, _ = self.sess.run([self.merged, self.final_ops], feed_dict=td_map)
        if global_step % 10 == 0:
            self.writer.add_summary(summary, global_step)

    def save(self):
        print('Start Saving')
        for part in self.variable_save_optimize_parts:
            part.save(self.sess)
        print('Saving Done.')
        