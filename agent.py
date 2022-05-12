from collections import OrderedDict
import time, os
curdir = os.path.expanduser("~")
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
tf.keras.backend.set_epsilon(tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps) # 1e-7 default
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_util, model_util as util, model_nets as nets
# CUDA 11.8.0_522.06, CUDNN 8.6.0.163, tensorflow-gpu==2.10.0, tensorflow_probability==0.