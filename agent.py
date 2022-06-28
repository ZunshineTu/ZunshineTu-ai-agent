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

        self.arch, self.env, self.trader, self.env_render, self.save_model = arch, env, trader, env_render, save_model
        self.chkpts, self.max_episodes, self.max_steps, self.learn_rates = tf.constant(chkpts, tf.int32), tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), {}
        for k,v in learn_rates.items(): self.learn_rates[k] = tf.constant(v, compute_dtype)
        self.initializer = tf.keras.initializers.GlorotUniform(time.time_ns())

        self.obs_spec, self.obs_zero, _ = gym_util.get_spec(env.observation_space, space_name='obs', compute_dtype=self.compute_dtype, net_attn_io=net_attn['io'], aio_max_latents=aio_max_latents, mixture_multi=mixture_multi)
        self.action_spec, _, self.action_zero_out = gym_util.get_spec(env.action_space, space_name='actions', compute_dtype=self.compute_dtype, mixture_multi=mixture_multi)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.action_total_size = tf.constant(np.sum([np.prod(feat['step_shape']) for feat in self.action_spec]),compute_dtype)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1)), tf.TensorShape((1,2)) if trader else tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool, tf.float64]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)
        self.step_zero, self.step_size_one = tf.constant([[0]]), tf.constant([[1]])

        latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents}
        latent_spec.update({'inp':512, 'midp':256, 'outp':512, 'evo':64})
        if latent_dist == 'd': latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)}) # deterministic
        if latent_dist == 'c': latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)}) # categorical # TODO https://keras.io/examples/generative/vq_vae/
        if latent_dist == 'mx': latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous

        if aug_data_step: self.obs_spec += [{'space_name':'step', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        self.obs_spec += [{'space_name':'reward_prev', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        inputs = {'obs':self.obs_zero, 'step':[self.step_zero], 'reward_prev':[self.rewards_zero], 'return_goal':[self.rewards_zero]}


        if arch in ('PG',):
            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps}]
            stats_spec = [{'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}, {'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}, {'name':'delta', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            self.action = nets.ArchFull('A', inputs, opt_spec, stats_spec, self.obs_spec, self.action_spec, latent_spec, obs_latent=False, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, latent_multi=1, aug_data_pos=aug_data_pos)
            outputs = self.action(inputs)
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)


        self.metrics_spec()
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch); arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); setattr(self, arch, arch_run)


    def metrics_spec(self):
        metrics_loss = OrderedDict()
        metrics_loss['2rewards*'] = {'-rewards_ma':np.float64, '-rewards_total+':np.float64, 'rewards_final=':np.float64}
        metrics_loss['1steps'] = {'steps+':np.int64}
        if self.arch == 'PG':
            metrics_loss['1nets*'] = {'-loss_ma':np.float64, '-loss_action':np.float64}
            metrics_loss['1extras'] = {'loss_action_returns':np.float64}
            metrics_loss['1extras2*'] = {'actlog0':np.float64, 'actlog1':np.float64}

    