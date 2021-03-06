#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

import random
import tensorflow as tf
import numpy as np
import yaml
from tqdm import tqdm
import datetime
import functools
import sys
import os
import copy

from candy.srv import Step, Value, UpdateWeights
from std_msgs.msg import String
import msgpack
import msgpack_numpy as m
m.patch()
import rospy


import sys
if not (sys.version_info[0] < 3):
	print = functools.partial(print, flush=True)


class ARGS(object):
	pass

from machine import Machine


TRAIN_EPOCH = 30
BATCH_SIZE = 1
global_step = 0
batch = []


if __name__ == '__main__':
	rospy.init_node('trainer_candy')
	machine = Machine()

	def calculate_difficulty(reward, vaerecon):
		# return vaerecon * vaerecon * abs(reward)
		return vaerecon

	def memory_training(msg):
		obs, actions, values, neglogpacs, rewards, vaerecons, states, std_actions, manual = msgpack.unpackb(msg.data, raw=False, encoding='utf-8')

		global batch
		global global_step	
		if len(batch) > 300:
			batch = batch[:300]
		if global_step % (TRAIN_EPOCH * 10) == 0:
			batch = []
		l = len(obs)
		for i in range(12, l - 12):
			future_obs = obs[i+12]
			batch.append( (calculate_difficulty(rewards[i], vaerecons[i]), [obs[i], actions[i], values[i], neglogpacs[i], rewards[i], vaerecons[i], states[i], std_actions[i], manual[i], copy.deepcopy(future_obs)]) )
		# for i in range(l - 12, l):
		# 	future_obs = ( [np.zeros([160, 160, 12, 3]) for i in range(4)], 0.0, np.zeros([12, 2]) )
		# 	batch.append( (calculate_difficulty(rewards[i], vaerecons[i]), [obs[i], actions[i], values[i], neglogpacs[i], rewards[i], vaerecons[i], states[i], std_actions[i], manual[i], copy.deepcopy(future_obs)]) )

		# print(self.rewards)
		# print(self.values)
		# print(np.array(self.rewards) - np.array([i[0] for i in self.values]))
		batch = sorted(batch, key = lambda y: y[0], reverse=True)
		print('Difficulty !', [t[0] for t in batch[:5]])
		# difficulty = np.array(difficulty)
		# print(difficulty[-20:])
		# def softmax(x):
		# 	x = np.clip(x, 1e-5, 1e5)
		# 	return np.exp(x) / np.sum(np.exp(x), axis=0)
		# difficulty = softmax(difficulty * 50)
		# print(difficulty[-20:])

		print("Memory Extraction Done.")

		for _ in tqdm(range(TRAIN_EPOCH)):
			roll = np.random.choice(len(batch), BATCH_SIZE)
			tbatch = []
			for i in roll:
				tbatch.append(batch[i])
			tra_batch = [np.array([t[1][i] for t in tbatch]) for i in range(10)]
			# tra_batch = [np.array([t[i] for t in tbatch]) for i in range(7)]
			machine.train(tra_batch, global_step)
			global_step += 1

		machine.save()

		if random.randint(1,1) == 1:
			rospy.wait_for_service('update_weights')
			try:
				update_weights = rospy.ServiceProxy('update_weights', UpdateWeights)

				param = machine.sess.run(machine.params)
				# for each in tqdm(machine.params):
				# 	param.append(np.array(each.eval(session=machine.sess)))
				# param = np.array(param)
				outmsg = msgpack.packb(param, use_bin_type=True)
				print('Update weights called!')
				update_weights(outmsg)

			except rospy.ServiceException as e:
				print("Service call failed: %s" % e)

	sub = rospy.Subscriber('/train_data', String, memory_training, queue_size=1)
	rospy.spin()
