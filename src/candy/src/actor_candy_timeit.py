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

from modules.utils.utils import timeit

if __name__ == '__main__':
    rospy.init_node('actor_candy')
    machine = Machine()

    def step(data):
        obs, state = timeit(lambda: msgpack.unpackb(data.a, raw=False, encoding='utf-8'), 'step/unpackb')
        # print(np.array(obs).shape)
        # print(np.array(state).shape)
        a, b, c, d, e = timeit(lambda: machine.step(obs, state), 'machine.step')
        outmsg = timeit(lambda: msgpack.packb([a,b,c,d,e], use_bin_type=True), 'step/pack')
        return outmsg

    def value(data):
        obs, state, action = timeit(lambda: msgpack.unpackb(data.a, raw=False, encoding='utf-8'), 'value/unpackb')
        # print(np.array(obs).shape)
        # print(np.array(state).shape)
        # print(np.array(action).shape)
        a,b,c,d,e = timeit(lambda: machine.value(obs, state, action), 'machine.value')
        outmsg = timeit(lambda: msgpack.packb([a,b,c,d,e], use_bin_type=True), 'value/pack')
        return outmsg

    def update_weights(data):
        param = timeit(lambda: msgpack.unpackb(data.a, raw=False, encoding='utf-8'), 'update_weights/unpack')
        timeit(lambda: machine.update_weights(param), 'machine.update_weights')
        return ''
 
    _ = timeit(lambda: rospy.Service('model_step', Step, step), 'rospy.Service(model_step)')
    _ = timeit(lambda: rospy.Service('model_value', Value, value), 'rospy.Service(model_value)')
    _ = timeit(lambda: rospy.Service('update_weights', UpdateWeights, update_weights), 'rospy.Service(update_weights)')
    rospy.spin()