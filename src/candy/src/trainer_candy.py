#!/usr/bin/env python
from __future__ import print_function, absolute_import, division

import random
import tensorflow as tf
import numpy as np 
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
import random
from collections import deque
from modules.utils.utils import load_args
import sys
if not (sys.version_info[0] < 3):
	print = functools.partial(print, flush=True)

from machine import Machine


TRAIN_EPOCH = 50
BATCH_SIZE = load_args()['batch_size']
global_step = 0
MAX_BUFFER_SIZE = 1e4
buffer = deque(maxlen=MAX_BUFFER_SIZE)

if __name__ == '__main__':
	rospy.init_node('trainer_candy')
	machine = Machine()

	def calculate_difficulty(reward, vaerecon):
		# return vaerecon * vaerecon * abs(reward)
		return vaerecon

	def memory_training(msg):
		obs, actions, values, neglogpacs, rewards, vaerecons, states, std_actions, manual = msgpack.unpackb(msg.data, raw=False, encoding='utf-8')

		global buffer
		global global_step
	
		buffer.extend(zip(calculate_difficulty(rewards, vaerecons), zip(obs, actions, values, neglogpacs, rewards, vaerecons, states, std_actions, manual)))

		# I don't use priority queue here for efficiency, maybe it really worth a trial
		# buffer = sorted(buffer, key = lambda y: y[0], reverse=True)
		# print('Difficulty !', [t[0] for t in buffer[:5]])

		print("Memory Extraction Done.")

		for _ in tqdm(range(TRAIN_EPOCH)):
			batch = random.sample(buffer, BATCH_SIZE)
			info_len = len(batch[0][1])
			tra_batch = [np.array([t[1][i] for t in batch]) for i in range(info_len)]
			# tra_batch = [np.array([t[i] for t in tbatch]) for i in range(7)]
			machine.train(tra_batch, global_step)
			global_step += 1

		machine.save()

		rospy.wait_for_service('update_weights')
		try:
			update_weights = rospy.ServiceProxy('update_weights', UpdateWeights)

			param = machine.sess.run(machine.params)
			# for each in tqdm(machine.params):
			# 	param.append(np.array(each.eval(session=machine.sess)))
			# param = np.array(param)
			outmsg = msgpack.packb(param, use_bin_type=True)
			update_weights(outmsg)

		except rospy.ServiceException as e:
			print("Service call failed: %s" % e)

	sub = rospy.Subscriber('/train_data', String, memory_training, queue_size=1)
	rospy.spin()
