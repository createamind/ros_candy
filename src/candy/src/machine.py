#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

from modules.multimodal import MultiModal
from modules.ppo import PPO

import tensorflow as tf
import numpy as np
import yaml
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
from modules.losses import MSELoss

import sys
if not (sys.version_info[0] < 3):
    print = functools.partial(print, flush=True)

class Machine(object):
    def __init__(self):

        args = self.get_args()
        self.args = args

        #Building Graph
        self.is_training = tf.placeholder(tf.bool, shape=(None), name='is_training')
        self.multimodal_train = MultiModal(False, args, 'multimodal', is_training=self.is_training, reuse=False)
        self.multimodal_test = MultiModal(True, args, 'multimodal', is_training=self.is_training, reuse=True)

        self.speed = tf.placeholder(tf.float32, shape=(args['batch_size'], 1), name='speed')
        self.test_speed = tf.placeholder(tf.float32, shape=(1, 1), name='test_speed')

        z = self.multimodal_train.mean
        test_z = self.multimodal_test.mean

        # z = tf.concat([z[:,:15], self.speed], 1)
        # test_z = tf.concat([test_z[:,:15], self.test_speed], 1)

        z = tf.clip_by_value(z, -5, 5)
        test_z = tf.clip_by_value(test_z, -5, 5)

        self.ppo = PPO(args, 'ppo', z=z, test_z=test_z, ent_coef=0.00000001, vf_coef=1, max_grad_norm=0.5)

        # self.test_vae_loss.inference()
        # z = self.c3d_encoder.inference()

        self.variable_restore_parts = [self.multimodal_train, self.multimodal_test, self.ppo]
        self.variable_save_optimize_parts = [self.multimodal_train, self.ppo]

        total_loss = self.multimodal_train.loss + 10 * self.ppo.loss

        #Not Turn Quickly Loss:
        self.smooth_loss = MSELoss(self.ppo.train_model.a0[:,1], self.multimodal_train.actions[:,1], args, 'smooth_loss', is_training=self.is_training, reuse=False)
        total_loss += self.smooth_loss.outputs

        tf.summary.scalar('total_loss', tf.reduce_mean(total_loss))

        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)
    
        self.final_ops = []
        for part in self.variable_save_optimize_parts:
            self.final_ops.append(part.optimize(total_loss))
        self.final_ops = tf.group(self.final_ops)

        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True


        self.merged = tf.summary.merge_all()
        self.sess = tf.Session(config = config)
        self.writer = tf.summary.FileWriter('/tmp/logs/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + 'image', self.sess.graph)

        with tf.Graph().as_default() as g:
            tf.Graph.finalize(g)
        self.sess.run(tf.global_variables_initializer())

        print('Restoring!')

        for part in self.variable_restore_parts:
            part.variable_restore(self.sess)

        print('Get_Params!')
        self.params = []
        for va in tf.trainable_variables():
            self.params.append(va)
        print(len(self.params), 'params in all!')

        print('Model Started!')

    def get_args(self):
        with open(os.path.join(sys.path[0], "args.yaml"), 'r') as f:
            try:
                t = yaml.load(f)
                return t
            except yaml.YAMLError as exc:
                print(exc)

    def step(self, obs, state):
        # mask = np.zeros(1)
        td_map = {self.ppo.act_model.S:state}

        td_map[self.multimodal_test.camera_left] = np.array([obs[0][0]])
        td_map[self.multimodal_test.camera_right] = np.array([obs[0][1]])
        td_map[self.multimodal_test.eye_left] = np.array([obs[0][2]])
        td_map[self.multimodal_test.eye_right] = np.array([obs[0][3]])
        td_map[self.multimodal_test.actions] = np.array([obs[2]])

        # td_map[self.test_raw_image] = np.array([obs[0][1]])
        # td_map[self.test_raw_image] = np.array([obs[0][2]])

        td_map[self.is_training] = False
        td_map[self.test_speed] = np.array([[obs[1]]]) # speed
        # td_map[self.test_steer] = np.array([[obs[2]]])

        return self.sess.run([self.ppo.act_model.a0, self.ppo.act_model.v0, self.ppo.act_model.snew, self.ppo.act_model.neglogp0, self.multimodal_test.loss], td_map)


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
            part.variable_restore(self.sess)

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
        td_map[self.is_training] = True

        # mask = np.zeros(self.args['batch_size'])
        td_map[self.ppo.train_model.S] = np.squeeze(states, 1)
        # td_map[self.ppo.train_model.M] = mask

        td_map[self.ppo.std_action] = std_actions
        td_map[self.ppo.std_mask] = manual

        td_map[self.multimodal_train.camera_left] = np.array([ob[0][0] for ob in obs])
        td_map[self.multimodal_train.camera_right] = np.array([ob[0][1] for ob in obs])
        td_map[self.multimodal_train.eye_left] = np.array([ob[0][2] for ob in obs])
        td_map[self.multimodal_train.eye_right] = np.array([ob[0][3] for ob in obs])
        td_map[self.multimodal_train.actions] = np.array([ob[2] for ob in obs])

        td_map[self.speed] = np.array([[ob[1]] for ob in obs])

        td_map[self.multimodal_test.camera_left] = np.array([obs[0][0][0]])
        td_map[self.multimodal_test.camera_right] = np.array([obs[0][0][1]])
        td_map[self.multimodal_test.eye_left] = np.array([obs[0][0][2]])
        td_map[self.multimodal_test.eye_right] = np.array([obs[0][0][3]])
        td_map[self.multimodal_test.actions] = np.array([obs[0][2]])
        td_map[self.test_speed] = np.array([[obs[0][1]]]) # speed

        # td_map[self.test_raw_image] = np.array([obs[0][1]])
        # td_map[self.test_raw_image] = np.array([obs[0][2]])


        summary, _ = self.sess.run([self.merged, self.final_ops], feed_dict=td_map)
        if global_step % 10 == 0:
            self.writer.add_summary(summary, global_step)


    def save(self):
        print('Start Saving')
        for part in self.variable_save_optimize_parts:
            part.save(self.sess)
        print('Saving Done.')