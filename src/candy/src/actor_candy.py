#!/usr/bin/env python
from __future__ import print_function, absolute_import, division
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
import rospy

import sys
if not (sys.version_info[0] < 3):
    print = functools.partial(print, flush=True)

class ARGS(object):
    pass


from machine import Machine

from modules.utils.debug_tools import timeit

if __name__ == '__main__':
    rospy.init_node('actor_candy')
    machine = timeit(Machine, 'Machine Initialization')

    def step(data):
        obs, state = msgpack.unpackb(data.a, raw=False, encoding='utf-8')
        # print(np.array(obs).shape)
        # print(np.array(state).shape)
        a, b, c, d, e = machine.step(obs, state)
        outmsg = msgpack.packb([a,b,c,d,e], use_bin_type=True)
        return outmsg

    def value(data):
        obs, state, action = msgpack.unpackb(data.a, raw=False, encoding='utf-8')
        # print(np.array(obs).shape)
        # print(np.array(state).shape)
        # print(np.array(action).shape)
        a,b,c,d,e = machine.value(obs, state, action)
        outmsg = msgpack.packb([a,b,c,d,e], use_bin_type=True)
        return outmsg

    def update_weights(data):
        param = msgpack.unpackb(data.a, raw=False, encoding='utf-8')
        machine.update_weights(param)
        return ''
 
    _ = rospy.Service('model_step', Step, step)
    _ = rospy.Service('model_value', Value, value)
    _ = rospy.Service('update_weights', UpdateWeights, update_weights)
    rospy.spin()