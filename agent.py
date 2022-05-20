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
# CUDA 11.8.0_522.06, CUDNN 8.6.0.163, tensorflow-gpu==2.10.0, tensorflow_probability==0.18.0
physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, trader, env_render, save_model, chkpts, max_episodes, max_steps, learn_rates, latent_size, latent_dist, mixture_multi, net_lstm, net_attn, aio_max_latents, aug_data_step, aug_data_pos):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_max = tf.constant(compute_dtype.max, compute_dtype)
        self.float_maxroot = tf.constant(tf.math.sqrt(self.float_max), compute_dtype)
        self.float_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype)
        self.float_eps_max = tf.constant(1.0 / self.float_eps, compute_dtype)
        self.loss_scale = tf.math.exp(tf.math.log(self.float_eps_max) * (1/2))
        self.float64_zero = tf.constant(0, tf.float64)

        self.arch, self.env, self.trader, self.env_render, self.save_model = 