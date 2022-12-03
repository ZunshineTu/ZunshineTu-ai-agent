
from collections import OrderedDict
import time, os, keyboard # , talib, bottleneck
import multiprocessing as mp
curdir = os.path.expanduser("~")
from sys import platform
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0,1,2,3
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit' # lets XLA work on CPU
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
# tf.config.run_functions_eagerly(True)
# tf.config.optimizer.set_jit("autoclustering") # enable XLA
# tf.config.experimental.enable_mlir_graph_optimization()
# tf.random.set_seed(0) # TODO https://www.tensorflow.org/guide/random_numbers
tf.keras.backend.set_epsilon(tf.experimental.numpy.finfo(tf.keras.backend.floatx()).eps) # 1e-7 default
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gymnasium as gym #, gym_algorithmic, procgen, pybullet_envs
# gym.logger.set_level(gym.logger.DISABLED)
import gym_util, model_util as util, model_nets as nets

# CUDA 11.8.0_522.06, CUDNN 8.6.0.163, tensorflow-gpu==2.10.0, tensorflow_probability==0.18.0
physical_devices_gpu = tf.config.list_physical_devices('GPU')
for i in range(len(physical_devices_gpu)): tf.config.experimental.set_memory_growth(physical_devices_gpu[i], True)

# TODO add Fourier prior like PercieverIO or https://github.com/zongyi-li/fourier_neural_operator
# TODO add S4 layer https://github.com/HazyResearch/state-spaces
# TODO how does CLIP quantize latents? https://github.com/openai/CLIP

# TODO try out MuZero-ish architecture
# TODO add Perciever, maybe ReZero

# TODO add GenNet and DisNet for GAN type boost
# TODO put actor in seperate process so can run async
# TODO add ZMQ and latent pooling

# TODO how to incorporate ARS random policy search?
# TODO try out the 'lottery ticket hypothosis' pruning during training
# TODO use numba to make things faster on CPU


class GeneralAI(tf.keras.Model):
    def __init__(self, arch, env, trader, env_render, save_model, chkpts, max_episodes, pause_episodes, max_steps, learn_rates, value_cont, latent_size, latent_dist, mixture_multi, net_lstm, net_attn, aio_max_latents, attn_mem_base, aug_data_step, aug_data_pos):
        super(GeneralAI, self).__init__()
        compute_dtype = tf.dtypes.as_dtype(self.compute_dtype)
        self.float_max, self.float64_max = tf.constant(compute_dtype.max, compute_dtype), tf.constant(tf.float64.max, tf.float64)
        self.float_maxroot = tf.constant(tf.math.sqrt(self.float_max), compute_dtype); self.float_minroot = tf.constant(1.0 / self.float_maxroot, compute_dtype)
        self.float_eps, self.float64_eps = tf.constant(tf.experimental.numpy.finfo(compute_dtype).eps, compute_dtype), tf.constant(tf.experimental.numpy.finfo(tf.float64).eps, tf.float64)
        self.float_eps_max = tf.constant(1.0 / self.float_eps, compute_dtype)
        self.loss_scale = tf.math.exp(tf.math.log(self.float_eps_max) * (1/2))
        self.compute_zero, self.float64_zero = tf.constant(0, compute_dtype), tf.constant(0, tf.float64)
        self.int32_max, self.int32_zero, self.int32_maxbit = tf.constant(tf.int32.max, tf.int32), tf.constant(0, tf.int32), tf.constant(1073741824, tf.int32)
        self.max_softmax = tf.constant(15, compute_dtype) if compute_dtype == tf.float64 else tf.constant(5, compute_dtype)

        self.arch, self.env, self.trader, self.env_render, self.save_model, self.value_cont = arch, env, trader, env_render, save_model, value_cont
        self.chkpts, self.max_episodes, self.max_steps, self.attn_mem_base, self.learn_rates = tf.constant(chkpts, tf.int32), tf.constant(max_episodes, tf.int32), tf.constant(max_steps, tf.int32), tf.constant(attn_mem_base, tf.int32), {}
        for k,v in learn_rates.items(): self.learn_rates[k] = tf.constant(v, compute_dtype)
        self.dist_prior = tfp.distributions.Independent(tfp.distributions.Logistic(loc=tf.zeros(latent_size, dtype=self.compute_dtype), scale=10.0), reinterpreted_batch_ndims=1)
        # self.dist_prior = tfp.distributions.Independent(tfp.distributions.Uniform(low=tf.cast(tf.fill(latent_size,-10), dtype=self.compute_dtype), high=10), reinterpreted_batch_ndims=1)
        self.initializer = tf.keras.initializers.GlorotUniform(time.time_ns())
        # self.initializer = tf.keras.initializers.GlorotNormal(time.time_ns())

        self.obs_spec, self.obs_zero, _ = gym_util.get_spec(env.observation_space, space_name='obs', compute_dtype=self.compute_dtype, net_attn_io=net_attn['io'], aio_max_latents=8, mixture_multi=mixture_multi)
        self.action_spec, _, self.action_zero_out = gym_util.get_spec(env.action_space, space_name='actions', compute_dtype=self.compute_dtype, mixture_multi=mixture_multi)
        self.obs_spec_len, self.action_spec_len = len(self.obs_spec), len(self.action_spec)
        self.action_total_size = tf.constant(np.sum([np.prod(feat['step_shape']) for feat in self.action_spec]),compute_dtype)
        self.gym_step_shapes = [feat['step_shape'] for feat in self.obs_spec] + [tf.TensorShape((1,1)), tf.TensorShape((1,1)), tf.TensorShape((1,2)) if trader else tf.TensorShape((1,1))]
        self.gym_step_dtypes = [feat['dtype'] for feat in self.obs_spec] + [tf.float64, tf.bool, tf.float64]
        self.rewards_zero, self.dones_zero = tf.constant([[0]],tf.float64), tf.constant([[False]],tf.bool)
        self.step_zero, self.step_size_one = tf.constant([[0]]), tf.constant([[1]])

        latent_spec = {'dtype':compute_dtype, 'latent_size':latent_size, 'num_latents':1, 'max_latents':aio_max_latents}
        # latent_spec.update({'inp':latent_size*4, 'midp':latent_size*2, 'outp':latent_size*4, 'evo':int(latent_size/2)})
        # latent_spec.update({'inp':512, 'midp':256, 'outp':512, 'evo':int(latent_size/2)})
        latent_spec.update({'inp':512, 'midp':256, 'outp':512, 'evo':64})
        if latent_dist == 'd': latent_spec.update({'dist_type':'d', 'num_components':latent_size, 'event_shape':(latent_size,)}) # deterministic
        if latent_dist == 'c': latent_spec.update({'dist_type':'c', 'num_components':0, 'event_shape':(latent_size, latent_size)}) # categorical # TODO https://keras.io/examples/generative/vq_vae/
        if latent_dist == 'mx': latent_spec.update({'dist_type':'mx', 'num_components':int(latent_size/16), 'event_shape':(latent_size,)}) # continuous

        if aug_data_step: self.obs_spec += [{'space_name':'step', 'name':'', 'dtype':tf.int64, 'dtype_out':compute_dtype, 'min':0, 'max':self.float_max, 'dist_type':'d', 'num_components':1, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        self.obs_spec += [{'space_name':'reward_prev', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':-self.float64_max, 'max':self.float64_max, 'dist_type':'d', 'num_components':1, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        # self.obs_spec += [{'space_name':'return_goal', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':-self.float64_max, 'max':self.float64_max, 'dist_type':'d', 'num_components':1, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]
        inputs = {'obs':self.obs_zero, 'step':[self.step_zero], 'reward_prev':[self.rewards_zero], 'return_goal':[self.rewards_zero]}

        if arch in ('MU',):
            self.mem_img_size = 4 # int(max_steps/4)
            self.obs_spec += [{'space_name':'done_prev', 'name':'', 'dtype':tf.bool, 'dtype_out':tf.int32, 'min':0, 'max':1, 'dist_type':'c', 'num_components':2, 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]; inputs['done_prev'] = [self.dones_zero]
            self.obs_spec += self.action_spec; inputs['actions'] = self.action_zero_out
            # self.obs_spec += [{'space_name':'return_goal', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]; inputs['return_goal'] = [self.rewards_zero]
            # opt_spec = [{'name':'rep', 'type':'a', 'schedule_type':'ep', 'num_steps':1000*max_steps, 'lr_min':tf.constant(3e-16, tf.float64), 'learn_rate':self.learn_rates['rep'], 'float_eps':self.float_eps}]
            opt_spec = [
                {'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['rep_action'], 'float_eps':self.float_eps},
                {'name':'trans', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['rep_trans'], 'float_eps':self.float_eps},
                {'name':'value', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['rep_value'], 'float_eps':self.float_eps},
                # {'name':'gen', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['rep_gen'], 'float_eps':self.float_eps},
            ]
            self.rep = nets.ArchTrans('RN', inputs, opt_spec, [], self.obs_spec, latent_spec, obs_latent=False, net_blocks=0, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=None, aug_data_pos=aug_data_pos); outputs = self.rep(inputs)
            # self.rep.optimizer_weights = util.optimizer_build(self.rep.optimizer['rep'], self.rep.trainable_variables)
            self.rep.optimizer_weights = []
            for spec in opt_spec: self.rep.optimizer_weights += util.optimizer_build(self.rep.optimizer[spec['name']], self.rep.trainable_variables)
            util.net_build(self.rep, self.initializer)
            rep_dist = self.rep.dist(outputs); self.latent_zero = tf.zeros_like(rep_dist.sample(), dtype=latent_spec['dtype'])
            latent_spec.update({'step_shape':self.latent_zero.shape}); self.latent_spec = latent_spec

            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}, {'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}]
            self.action = nets.ArchGen('AN', self.latent_zero, opt_spec, stats_spec, self.action_spec, latent_spec, net_blocks=3, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, latent_multi=2); outputs = self.action(self.latent_zero)
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)

            opt_spec = [{'name':'trans', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['trans'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            latent_spec_trans = [latent_spec.copy()]; latent_spec_trans[0].update({'space_name':'latent_out', 'name':'', 'dtype_out':compute_dtype, 'dist_type':'d', 'num_components':1, 'event_shape':(latent_size,), 'event_size':int(np.prod(self.latent_zero.shape[1:-1]).item()), 'seq_size_out':self.mem_img_size}) # continuous
            # self.trans = nets.ArchAR('AR', self.latent_zero, opt_spec, stats_spec, latent_spec, net_blocks=4, net_lstm=net_lstm, net_attn={'net':True, 'io':True, 'out':False, 'ar':True}, num_heads=4, memory_size=max_steps, mem_img_size=self.mem_img_size, latent_multi=2); outputs = self.trans(self.latent_zero)
            self.trans = nets.ArchGen('TN', self.latent_zero, opt_spec, stats_spec, latent_spec_trans, latent_spec, net_blocks=4, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, latent_multi=3); outputs = self.trans(self.latent_zero)
            self.trans.optimizer_weights = util.optimizer_build(self.trans.optimizer['trans'], self.trans.trainable_variables)
            util.net_build(self.trans, self.initializer)