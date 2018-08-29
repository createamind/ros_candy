#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Int16, String, Float32
from cv_bridge import CvBridge

import cv2
import numpy as np
import tensorflow as tf
import copy

from candy.srv import Step, Value, UpdateWeights

import msgpack
import msgpack_numpy as m
m.patch()

import argparse
import logging
import random
import time
import datetime
try:
	import pygame
	from pygame.locals import K_DOWN
	from pygame.locals import K_LEFT
	from pygame.locals import K_RIGHT
	from pygame.locals import K_SPACE
	from pygame.locals import K_UP
	from pygame.locals import K_a
	from pygame.locals import K_d
	from pygame.locals import K_p
	from pygame.locals import K_q
	from pygame.locals import K_r
	from pygame.locals import K_s
	from pygame.locals import K_w
	from pygame.locals import K_t
	from pygame.locals import K_m
	from pygame.locals import K_n
	from pygame.locals import K_1
	from pygame.locals import K_2
	from pygame.locals import K_3
	from pygame.locals import K_4
	from pygame.locals import K_5
	from pygame.locals import K_6
	from pygame.locals import K_7
	from pygame.locals import K_8
	from pygame.locals import K_9
	from pygame.locals import K_v
	from pygame.locals import K_c
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

WINDOW_WIDTH = 160
WINDOW_HEIGHT = 160
BUFFER_LIMIT = 100


class Carla_Wrapper(object):

	def __init__(self, gamma=0.99, lam=0.95, nlstm=10):
		self.global_step = 0
		self.nlstm = nlstm
		self.lam = lam
		self.gamma = gamma
		self.state = np.zeros((1, nlstm*2), dtype=np.float32)

		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual = [],[],[],[],[],[],[],[],[]
		self.last_frame = [[np.zeros([160,160,1,3]) for i in range(11)] for j in range(4)]
		self.last_control = [[0.0, 0.0] for i in range(11)]
		self.publisher = rospy.Publisher('/train_data', String, queue_size=1)


	def clear(self):
		self.state = np.zeros((1, self.nlstm*2), dtype=np.float32)
		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual = [],[],[],[],[],[],[],[],[]
		self.last_frame = [[np.zeros([160,160,1,3]) for i in range(11)] for j in range(4)]
		self.last_control = [[0.0, 0.0] for i in range(11)]

	def update_reward(self, cnt, obs, action, reward):
		l = len(self.obs)
		for t in range(l - cnt, l - 2):
			self.rewards[t] = self.rewards[t+1]
		if reward is None:
			self.rewards[l-1] = self.rewards[l-2]
		else:
			self.rewards[l-1] = reward
		self.rewards[l-1] *= 20
		for t in reversed(range(l - cnt, l - 1)):
			self.rewards[t] += self.lam * self.rewards[t+1]

	def post_process(self, inputs, cnt):

		obs, reward, action, _, _ = self.pre_process(inputs, refresh=False)
		self.update_reward(cnt, obs, action, reward)

		print(self.rewards[-20:])
		print('Start Memory Replay')
		self.memory_training()
		print('Memory Replay Done')
		time.sleep(1) # Time for sending train data


	def pre_process(self, inputs, refresh=False):

		images, control, reward, std_control, manual, speed = inputs
		frame = [None] * 4
		images = copy.deepcopy(images)
		# print(images[0])
		for i, _ in enumerate(images):
			images[i] = images[i].astype(np.float32) / 128 - 1
			nowframe = np.expand_dims(images[i], 2)
			frame[i] = np.concatenate(self.last_frame[i] + [nowframe], 2)
			if refresh == True:
				self.last_frame[i] = self.last_frame[i][1:] + [nowframe]
		obs = [copy.deepcopy(frame), copy.deepcopy(speed), np.array(self.last_control + [[0.0,0.0]])]
		return obs, reward, control, std_control, manual
		

	# def update(self, inputs):

	# 	obs, reward, action, std_action, manual = self.pre_process(inputs, refresh=True)

	# 	rospy.wait_for_service('model_value')
	# 	try:
	# 		model_value = rospy.ServiceProxy('model_value', Value)

	# 		msg_input = msgpack.packb([obs, self.state, action], use_bin_type=True)
	# 		msg_output = model_value(msg_input)
	# 		_, value, self.state, neglogpacs, vaerecon = msgpack.unpackb(msg_output.b, raw=False, encoding='utf-8')

	# 	except rospy.ServiceException as e:
	# 		print("Service call failed: %s" % e)
	# 		return


	# 	self.states.append(self.state)
	# 	self.obs.append(obs)
	# 	self.actions.append(action)
	# 	self.values.append(value)
	# 	self.neglogpacs.append(neglogpacs)
	# 	self.rewards.append(reward)
	# 	self.vaerecons.append(vaerecon)
	# 	self.std_actions.append(std_action)
	# 	self.manual.append(manual)

		# self.red_buffer.append(red)
		# self.manual_buffer.append(manual)

	def pretrain(self):
		raise NotImplementedError
		# if os.path.exists('obs/data'):
		# 	print('Start Pretraining!!')
		# 	with open('obs/data', 'rb') as fp:
		# 		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states = msgpack.load(fp, encoding='utf-8', raw=False)
		# 	print('Pretraining length = ', len(self.obs))
		# 	self.memory_training(pretrain=True)

	def calculate_difficulty(self, reward, vaerecon):
		# return abs(reward)
		return vaerecon
		
	def memory_training(self, pretrain=False):
		l = len(self.obs)
		train_batch = [self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual]
		msg_str = msgpack.packb(train_batch, use_bin_type=True)
		self.publisher.publish(msg_str)
		self.obs, self.actions, self.values, self.neglogpacs, self.rewards, self.vaerecons, self.states, self.std_actions, self.manual = [],[],[],[],[],[],[],[],[]

	def get_control_and_update(self, inputs):

		obs, reward, _, std_action, manual = self.pre_process(inputs, refresh=True)

		rospy.wait_for_service('model_step')
		try:
			model_step = rospy.ServiceProxy('model_step', Step)
			msg_input = msgpack.packb([obs, self.state], use_bin_type=True)
			msg_output = model_step(msg_input)
			action, value, self.state, neglogpacs, vaerecon = msgpack.unpackb(msg_output.b, raw=False, encoding='utf-8')
			# print(action)
		except rospy.ServiceException as e:
			print("Service call failed: %s" % e)
			return [0,0]

		self.states.append(self.state)

		# Change obs
		if manual:
			obs[2] = np.array(self.last_control + [std_action])
			self.last_control = self.last_control[1:] + [std_action]
		else:
			obs[2] = np.array(self.last_control + [action[0]])
			self.last_control = self.last_control[1:] + [action[0]]

		self.obs.append(obs)
		self.actions.append(action)
		self.values.append(value)
		self.neglogpacs.append(neglogpacs)
		self.rewards.append(reward)
		self.vaerecons.append(vaerecon)
		self.std_actions.append(std_action)
		self.manual.append(manual)

		return action
	# def get_control(self, inputs):

	# 	obs, _, _, _, manual = self.pre_process(inputs)
	# 	rospy.wait_for_service('model_step')

	# 	try:
	# 		model_step = rospy.ServiceProxy('model_step', Step)
	# 		msg_input = msgpack.packb([obs, self.state], use_bin_type=True)
	# 		msg_output = model_step(msg_input)
	# 		action, _, _, _, _ = msgpack.unpackb(msg_output.b, raw=False, encoding='utf-8')
	# 		# print(action)
	# 		return action
	# 	except rospy.ServiceException as e:
	# 		print("Service call failed: %s" % e)

	# 	return 0 # do nothing

class CarlaGame(object):
	def __init__(self, carla_wrapper, image_getter, image2_getter, lidar_getter, eyeleft_getter,
	 eyeright_getter, eyeback_getter, eyefront_getter, speed_getter, steer_getter, 
	 brake_throttle_getter, is_auto_getter, throttle_publisher, steer_publisher, train=False):
		self._timer = None
		self._display = None
		self.images = None
		self.should_display = True
		random.seed(datetime.datetime.now())
		self.manual = True
		self.cnt = 0
		self.endnow = False
		self.canreplay = train
		self.history_manual = False
		self.history_steer = 0.0
		pygame.init()

		self.image_getter = image_getter
		self.image2_getter = image2_getter
		# self.lidar_getter = lidar_getter
		self.eyeleft_getter = eyeleft_getter
		self.eyeright_getter = eyeright_getter
		# self.eyeback_getter = eyeback_getter
		# self.eyefront_getter = eyefront_getter


		self.throttle_publisher = throttle_publisher
		self.steer_publisher = steer_publisher
		self.carla_wrapper = carla_wrapper

		self.speed_getter = speed_getter
		self.steer_getter = steer_getter
		self.brake_throttle_getter = brake_throttle_getter
		self.is_auto_getter = is_auto_getter

		self._display = pygame.display.set_mode(
			(WINDOW_WIDTH * 2, WINDOW_HEIGHT * 2),
			pygame.HWSURFACE | pygame.DOUBLEBUF)

	def execute(self):
		self._on_loop()
		self._on_render()
		pygame.event.pump() # process event queue


	def _calculate_reward(self, manual, history_manual, steer, history_steer):
		reward = 0
		if history_manual == False and manual == True:
			reward = -1
		reward = 0.1 - 20 * abs(history_steer - steer)
		return reward

	def _on_loop(self):
		left_image = self.image_getter()
		right_image = self.image2_getter()
		eyeleft = self.eyeleft_getter()
		eyeright = self.eyeright_getter()
		if left_image is None or right_image is None or eyeleft is None or eyeright is None:
			return

		
		speed = self.speed_getter()
		steer = self.steer_getter()
		brake_throttle = self.brake_throttle_getter()
		manual = not self.is_auto_getter()


		control, reward = self._get_keyboard_control(pygame.key.get_pressed())
		if reward is None:
			reward = self._calculate_reward(manual, self.history_manual, steer, self.history_steer)
		if control == "done":
			return
		elif control is None:
			return

		self.history_manual = manual
		self.history_steer = steer

		control = [brake_throttle, steer]
		self.images = [left_image, right_image, eyeleft, eyeright]

		self.cnt += 1
		model_control = self.carla_wrapper.get_control_and_update([self.images, control, reward, control, manual, speed])

		if len(np.array(model_control).shape) != 1:
			model_control = model_control[0]
		print("throttle=%.2f, %.2f ---- steer=%.2f, %.2f" % (control[0], model_control[0], control[1], model_control[1]))
		if manual:
			print("Human!")
		else:
			print("Model!")

		# if self.manual_control:
		# 	self.throttle_publisher.publish(max(-1.0, min(1.0, control[0])))
		# 	self.steer_publisher.publish(max(-1.0, min(1.0, control[1])))
		# else:
		self.throttle_publisher.publish(max(-1.0, min(1.0, model_control[0])))
		self.steer_publisher.publish(max(-1.0, min(1.0, model_control[1])))

		if self.endnow or self.cnt > BUFFER_LIMIT:
			if self.canreplay:
				self.carla_wrapper.post_process([self.images, model_control, reward, control, manual, speed], self.cnt)
			self.carla_wrapper.clear()
			self.cnt = 0
		self.endnow = False
		
	def _get_keyboard_control(self, keys):
		th = 0
		steer = 0

		if keys[K_r]:
			return None, None
		if keys[K_t]:
			self.should_display = not self.should_display
			return 'done', None
		if keys[K_m]:
			self.manual = True
			return 'done', None
		if keys[K_n]:
			self.manual = False
			return 'done', None
		if keys[K_v]:
			self.endnow = True
			return 'done', None

		if keys[K_LEFT] or keys[K_a]:
			steer = -1
		if keys[K_RIGHT] or keys[K_d]:
			steer = 1
		if keys[K_UP] or keys[K_w]:
			th = 1
		if keys[K_DOWN] or keys[K_s]:
			th = -1

		if keys[K_c]:
			self.canreplay = not self.canreplay

		reward = None
		if keys[K_1]:
			reward = -1
		if keys[K_2]:
			reward = -0.5
		if keys[K_3]:
			reward = -0.25
		if keys[K_4]:
			reward = -0.1
		if keys[K_5]:
			reward = 0
		if keys[K_6]:
			reward = 0.1
		if keys[K_7]:
			reward = 0.25
		if keys[K_8]:
			reward = 0.5
		if keys[K_9]:
			reward = 1

		# cod = 0
		# if steer == -1 and th == 1:
		# 	cod = 1
		# elif steer == 0 and th == 1:
		# 	cod = 2
		# elif steer == 1 and th == 1:
		# 	cod = 3
		# elif steer == -1 and th == 0:
		# 	cod = 4
		# elif steer == 1 and th == 0:
		# 	cod = 5
		# elif steer == -1 and th == -1:
		# 	cod = 6
		# elif steer == 0 and th == -1:
		# 	cod = 7
		# elif steer == 1 and th == -1:
		# 	cod = 8
			
		return [th, steer], reward

	def _on_render(self):
		if self.should_display == False or self.images is None:
			return
		left_image, right_image, eyeleft, eyeright = self.images
		if left_image is not None:
			array = left_image
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (0, 0))

		if right_image is not None:
			array = right_image
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (WINDOW_WIDTH, 0))

		if eyeleft is not None:
			array = eyeleft
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (0, WINDOW_HEIGHT))

		if eyeright is not None:
			array = eyeright
			surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
			self._display.blit(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))

		pygame.display.flip()



class WrapperCandy():
	def __init__(self, load_mode = False):
		self._cv_bridge = CvBridge()

		self._sub = rospy.Subscriber('/stereo/left/image_raw', Image, self.load_image, queue_size=1)
		self._subb = rospy.Subscriber('/stereo/right/image_raw', Image, self.load_image2, queue_size=1)
		# self._subb1 = rospy.Subscriber('/rslidar_points', PointCloud2, self.load_lidar, queue_size=1)
		self._subb2 = rospy.Subscriber('/multiple/webcam3/image_raw', Image, self.load_eyeleft, queue_size=1)
		self._subb3 = rospy.Subscriber('/multiple/webcam1/image_raw', Image, self.load_eyeright, queue_size=1)
		# self._subb4 = rospy.Subscriber('/multiple/webcam0/image_raw', Image, self.load_eyeback, queue_size=1)
		# self._subb5 = rospy.Subscriber('/multiple/webcam2/image_raw', Image, self.load_eyefront, queue_size=1)

		self._sub2 = rospy.Subscriber('/current_speed', Float32, self.load_speed, queue_size=1)
		self._sub3 = rospy.Subscriber('/current_steer', Float32, self.load_steer, queue_size=1)
		self._sub4 = rospy.Subscriber('/current_brake_throttle', Float32, self.load_brake_throttle, queue_size=1)
		self._sub5 = rospy.Subscriber('/current_is_auto', Int16, self.load_is_auto, queue_size=1)

		self.throttle_publisher = rospy.Publisher('/ferrari_throttle', Float32, queue_size=1)
		self.steer_publisher = rospy.Publisher('/ferrari_steer', Float32, queue_size=1)

		self.all_pub = rospy.Publisher('/wrapper_data', String, queue_size=1)
		if load_mode:
			self.all_sub = rospy.Subscriber('/wrapper_data', String, self.all_loader, queue_size=1)

		self.image = None
		self.image2 = None
		self.lidar = None
		self.eyeleft = None
		self.eyeright = None
		self.eyeback = None
		self.eyefront = None

		self.speed = 0
		self.steer = 0
		self.is_auto = True
		self.brake_throttle = 0

	def image_getter(self):
		def func():
			if self.image is None:
				return None
			image = self.image[130:, :]
			image = cv2.resize(image, (160,160))
			return image		
		return func
	def image2_getter(self):
		def func():
			if self.image2 is None:
				return None
			image = self.image2[130:, :]
			image = cv2.resize(image, (160,160))
			return image
		return func
	def lidar_getter(self):
		def func():
			return self.lidar
		return func
	def eyeleft_getter(self):
		def func():
			if self.eyeleft is None:
				return None
			image = self.eyeleft[130:, :]
			image = cv2.resize(image, (160,160))
			return image
		return func
	def eyeright_getter(self):
		def func():
			if self.eyeright is None:
				return None
			image = self.eyeright[130:, :]
			image = cv2.resize(image, (160,160))
			return image
		return func

	def eyeback_getter(self):
		def func():
			return self.eyeback
		return func

	def eyefront_getter(self):
		def func():
			return self.eyefront
		return func

	def speed_getter(self):
		def func():
			return self.speed / 15.0 - 1
		return func

	def steer_getter(self):
		def func():
			return self.steer
		return func

	def brake_throttle_getter(self):
		def func():
			return self.brake_throttle
		return func

	def is_auto_getter(self):
		def func():
			return self.is_auto
		return func

	def load_speed(self, msg):
		self.speed = msg.data

	def load_is_auto(self, msg):
		self.is_auto = False if msg.data == 0 else True

	def load_steer(self, msg):
		self.steer = max(min(1.0, msg.data),-1.0)

	def load_brake_throttle(self, msg):
		self.brake_throttle = max(min(1.0, msg.data),-1.0)

	def load_image(self, image_msg):
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		cv_image = cv2.resize(cv_image,(320,320))
		cv_image = cv2.flip(cv_image, -1)
		# print(cv_image)
		image = cv_image[...,::-1]
		self.image = image

	def load_image2(self, image_msg):
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		cv_image = cv2.resize(cv_image,(320,320))
		cv_image = cv2.flip(cv_image, -1)
		# print(cv_image)
		image = cv_image[...,::-1]
		self.image2 = image

	def load_eyeleft(self, image_msg):
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		cv_image = cv2.resize(cv_image,(320,320))
		cv_image = cv2.flip(cv_image, -1)
		# print(cv_image)
		image = cv_image[...,::-1]
		self.eyeleft = image

	def load_eyeright(self, image_msg):
		cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		cv_image = cv2.resize(cv_image,(320,320))
		cv_image = cv2.flip(cv_image, -1)
		# print(cv_image)
		image = cv_image[...,::-1]
		self.eyeright = image

	def load_eyeback(self, image_msg):
		# cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		# cv_image = cv2.resize(cv_image,(320,320))
		# cv_image = cv2.flip(cv_image, -1)
		# # print(cv_image)
		# image = cv_image[...,::-1]
		self.eyeback = None
		return

 	def load_eyefront(self, image_msg):
		# cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
		# cv_image = cv2.resize(cv_image,(320,320))
		# cv_image = cv2.flip(cv_image, -1)
		# # print(cv_image)
		# image = cv_image[...,::-1]
		self.eyefront = None
		return

 	def load_lidar(self, ros_cloud):
		self.lidar = None
		return
		# points_list = []
		# for data in pc2.read_points(ros_cloud, skip_nans=True):
		# 	points_list.append([data[0], data[1], data[2], data[3]])
		# self.lidar = points_list

	def train_image_load(self):
		PATH = '/data/forvae'
		import os
		from glob import glob
		result = sorted([y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.jpg'))])
		print(len(result))
		for _ in range(1000000):
			for v in result:
				image = cv2.imread(v)
				image = cv2.resize(image,(160,160))
				image = image[...,::-1]
				yield image

	def all_loader(self, msg):
		# self.image, self.image2, self.lidar, self.eyeleft, self.eyeright, self.speed, self.steer, self.is_auto, self.brake_throttle = msgpack.unpackb(msg.data, raw=False, encoding='utf-8')
		# self.image, self.speed, self.steer, self.is_auto, self.brake_throttle = msgpack.unpackb(msg.data, raw=False, encoding='utf-8')
		self.image, self.image2, self.eyeleft, self.eyeright, self.speed, self.steer, self.is_auto, self.brake_throttle = msgpack.unpackb(msg.data, raw=False, encoding='utf-8')

	def all_publisher(self):
		msg = msgpack.packb([self.image, self.image2, self.eyeleft, self.eyeright, self.speed, self.steer, self.is_auto, self.brake_throttle], use_bin_type=True)
		self.all_pub.publish(msg)

if __name__ == '__main__':
	argparser = argparse.ArgumentParser('Wrapper')
	argparser.add_argument(
		'-l', '--load-rosbag-data',
		action='store_true',
		help='load rosbag data')
	argparser.add_argument(
		'-t', '--train',
		action='store_true',
		help='training')
	argparser.add_argument(
		'-n', '--no-actor',
		action='store_true',
		help='no actor')
	args = argparser.parse_args()

	rospy.init_node('wrapper_candy')
	wrapper_candy = WrapperCandy(args.load_rosbag_data)
	carla_wrapper = Carla_Wrapper()


	carla_game = CarlaGame(carla_wrapper, wrapper_candy.image_getter(), wrapper_candy.image2_getter(), wrapper_candy.lidar_getter(), 
	wrapper_candy.eyeleft_getter(), wrapper_candy.eyeright_getter(), wrapper_candy.eyeback_getter(), wrapper_candy.eyefront_getter(), 
	wrapper_candy.speed_getter(), wrapper_candy.steer_getter(), wrapper_candy.brake_throttle_getter(), wrapper_candy.is_auto_getter(), 
	wrapper_candy.throttle_publisher, wrapper_candy.steer_publisher, args.train)


	rate = rospy.Rate(10) # 10hz
	# image_loader = wrapper_candy.train_image_load()
	while not rospy.is_shutdown():
		# wrapper_candy.image = image_loader.next()
		# print(wrapper_candy.image.shape)
		# print('speed', wrapper_candy.speed, 'steer', wrapper_candy.steer, 'is_auto', wrapper_candy.is_auto, 'brake_th', wrapper_candy.brake_throttle)
		if not args.load_rosbag_data:
			wrapper_candy.all_publisher()
		if not args.no_actor:
			carla_game.execute()
		rate.sleep()
