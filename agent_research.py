
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

            inputs_cond = {'obs':self.latent_zero, 'actions':self.action_zero_out}
            opt_spec = [{'name':'value', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['value'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            val_dist_type = ('mx',8) if value_cont else ('d',1)
            value_spec = [{'space_name':'value', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'dist_type':val_dist_type[0], 'num_components':val_dist_type[1], 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            value_spec += [{'space_name':'reward', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'dist_type':val_dist_type[0], 'num_components':val_dist_type[1], 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            value_spec += [{'space_name':'done', 'name':'', 'dtype':tf.bool, 'dtype_out':compute_dtype, 'dist_type':'c', 'num_components':2, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            # self.value = nets.ArchGen('VN', self.latent_zero, opt_spec, stats_spec, value_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps); outputs = self.value(self.latent_zero)
            self.value = nets.ArchFull('VL', inputs_cond, opt_spec, stats_spec, self.action_spec, value_spec, latent_spec, obs_latent=True, net_blocks=4, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, latent_multi=3, aug_data_pos=aug_data_pos); outputs = self.value(inputs_cond) # _val-cond
            self.value.optimizer_weights = util.optimizer_build(self.value.optimizer['value'], self.value.trainable_variables)
            util.net_build(self.value, self.initializer)

            # opt_spec = [{'name':'gen', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['gen'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            # # gen_spec = self.obs_spec
            # gen_spec = [{'space_name':'gen', 'name':'', 'dtype':compute_dtype, 'dtype_out':compute_dtype, 'dist_type':'d', 'num_components':1, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            # self.gen_spec_len = len(gen_spec)
            # self.gen = nets.ArchGen('GN', self.latent_zero, opt_spec, stats_spec, gen_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=None, latent_multi=1); outputs = self.gen(self.latent_zero)
            # self.gen.optimizer_weights = util.optimizer_build(self.gen.optimizer['gen'], self.gen.trainable_variables)
            # util.net_build(self.gen, self.initializer)

            # opt_spec = [{'name':'act', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['act'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}, {'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}]
            # # act_spec = [{'space_name':'return_goal', 'name':'', 'event_shape':(1,), 'event_size':1, 'channels':1, 'step_shape':tf.TensorShape((1,1)), 'num_latents':1}]; inputs_cond = {'obs':self.latent_zero, 'return_goal':[self.rewards_zero]}
            # # self.act = nets.ArchFull('ACT', inputs_cond, opt_spec, stats_spec, act_spec, self.action_spec, latent_spec, obs_latent=True, net_blocks=3, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos); outputs = self.act(inputs_cond)
            # self.act = nets.ArchGen('ACT', self.latent_zero, opt_spec, stats_spec, self.action_spec, latent_spec, net_blocks=3, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps); outputs = self.act(self.latent_zero)
            # self.act.optimizer_weights = util.optimizer_build(self.act.optimizer['act'], self.act.trainable_variables)
            # util.net_build(self.act, self.initializer)

            # opt_spec = [{'name':'pool', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['pool'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
            # self.pool = nets.ArchNet('LP', self.latent_zero, opt_spec, stats_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn={'net':True, 'io':True, 'out':False, 'ar':True}, num_heads=4, memory_size=max_steps, mem_img_size=self.mem_img_size); outputs = self.pool(self.latent_zero)
            # self.pool.optimizer_weights = util.optimizer_build(self.pool.optimizer['pool'], self.pool.trainable_variables)
            # util.net_build(self.pool, self.initializer)

        if arch in ('PG',):
            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}, {'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}]
            self.action = nets.ArchFull('A', inputs, opt_spec, stats_spec, self.obs_spec, self.action_spec, latent_spec, obs_latent=False, net_blocks=3, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, latent_multi=1, aug_data_pos=aug_data_pos); outputs = self.action(inputs)
            # inputs = {'obs':[self.obs_zero[0]], 'step':[self.step_zero], 'reward_prev':[self.rewards_zero], 'return_goal':[self.rewards_zero]} # PG shkspr img tests
            # self.action = nets.ArchFull('A', inputs, opt_spec, stats_spec, self.obs_spec[0:1]+self.obs_spec[2:], self.action_spec, latent_spec, obs_latent=False, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos); outputs = self.action(inputs) # PG shkspr img tests
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)
            # thresh = [2e-5,2e-3]; thresh_rates = [77,57,44] # 2e-12 107, 2e-10 89, 2e-8 71, 2e-6 53, 2e-5 44, 2e-4 35, 2e-3 26, 2e-2 17 # _lr-loss
            # thresh = [2e-5,2e-3]; thresh_rates = [77,57,44] # _lr-rwd-std
            # self.action_get_learn_rate = util.LearnRateThresh(thresh, thresh_rates)

        if arch in ('AC',):
            opt_spec = [
                {'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps},
                {'name':'value', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['value'], 'float_eps':self.float_eps},
            ]
            self.rep = nets.ArchTrans('RN', inputs, opt_spec, [], self.obs_spec, latent_spec, obs_latent=False, net_blocks=0, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=None, aug_data_pos=aug_data_pos); outputs = self.rep(inputs)
            self.rep.optimizer_weights = []
            for spec in opt_spec: self.rep.optimizer_weights += util.optimizer_build(self.rep.optimizer[spec['name']], self.rep.trainable_variables)
            util.net_build(self.rep, self.initializer)
            rep_dist = self.rep.dist(outputs); self.latent_zero = tf.zeros_like(rep_dist.sample(), dtype=latent_spec['dtype'])
            latent_spec.update({'step_shape':self.latent_zero.shape}); self.latent_spec = latent_spec

            opt_spec = [{'name':'action', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['action'], 'float_eps':self.float_eps}]; stats_spec = [{'name':'rwd', 'b1':0.99, 'b2':0.99, 'dtype':tf.float64}]
            self.action = nets.ArchGen('AN', self.latent_zero, opt_spec, stats_spec, self.action_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps); outputs = self.action(self.latent_zero)
            self.action.optimizer_weights = util.optimizer_build(self.action.optimizer['action'], self.action.trainable_variables)
            util.net_build(self.action, self.initializer)

            inputs_cond = {'obs':self.latent_zero, 'actions':self.action_zero_out}
            opt_spec = [{'name':'value', 'type':'a', 'schedule_type':'', 'learn_rate':self.learn_rates['value'], 'float_eps':self.float_eps}]
            if value_cont: value_spec = [{'space_name':'values', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'dist_type':'mx', 'num_components':8, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            else: value_spec = [{'space_name':'values', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'dist_type':'d', 'num_components':1, 'event_shape':(1,), 'event_size':1, 'step_shape':tf.TensorShape((1,1))}]
            # self.value = nets.ArchGen('VN', self.latent_zero, opt_spec, [], value_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps); outputs = self.value(self.latent_zero)
            self.value = nets.ArchFull('VN', inputs_cond, opt_spec, [], self.action_spec, value_spec, latent_spec, obs_latent=True, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn, num_heads=4, memory_size=max_steps, aug_data_pos=aug_data_pos); outputs = self.value(inputs_cond) # _val-cond
            self.value.optimizer_weights = util.optimizer_build(self.value.optimizer['value'], self.value.trainable_variables)
            util.net_build(self.value, self.initializer)

        # opt_spec = [{'name':'meta', 'type':'a', 'schedule_type':'', 'learn_rate':tf.constant(2e-5, tf.float64), 'float_eps':self.float_eps}]; stats_spec = [{'name':'loss', 'b1':0.99, 'b2':0.99, 'dtype':compute_dtype}]
        # inputs_meta = {'obs':[tf.constant([[0,0,0]],compute_dtype)]}; meta_spec_in = [{'space_name':'obs', 'name':'', 'event_shape':(3,), 'event_size':1, 'channels':3, 'step_shape':tf.TensorShape((1,3)), 'num_latents':1}]
        # self.meta_spec = [{'space_name':'meta', 'name':'', 'dtype':tf.float64, 'dtype_out':compute_dtype, 'min':self.float_eps, 'max':self.learn_rates['action'], 'dist_type':'mx', 'num_components':8, 'event_shape':(1,), 'step_shape':tf.TensorShape((1,1))}]
        # self.meta = nets.ArchFull('M', inputs_meta, opt_spec, stats_spec, meta_spec_in, self.meta_spec, latent_spec, net_blocks=2, net_lstm=net_lstm, net_attn=net_attn); outputs = self.meta(inputs_meta)
        # self.meta.optimizer_weights = util.optimizer_build(self.meta.optimizer['meta'], self.meta.trainable_variables)
        # util.net_build(self.meta, self.initializer)


        self.stop, self.stopped_episode, self.pause_episodes = False, max_episodes-1, pause_episodes
        if platform == "win32": keyboard.add_hotkey('ctrl+alt+k', self.on_stop, suppress=True) # TODO figure out linux/Docker version of this that works
        self.metrics_spec()
        # TF bug that wont set graph options with tf.function decorator inside a class
        self.reset_states = tf.function(self.reset_states, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
        self.reset_states()
        arch_run = getattr(self, arch); arch_run = tf.function(arch_run, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS); setattr(self, arch, arch_run)


    def metrics_spec(self):
        metrics_loss = OrderedDict()
        metrics_loss['2rewards*'] = {'-rewards_ma':np.float64, '-rewards_total+':np.float64, 'rewards_final=':np.float64}
        metrics_loss['1steps'] = {'steps+':np.int64}
        if self.arch == 'PG':
            metrics_loss['1~nets*'] = {'-loss_ma':np.float64, '-loss_action':np.float64}
            # metrics_loss['1extras'] = {'returns':np.float64}
            metrics_loss['1extrasAR'] = {'loss_action_returns':np.float64}
            metrics_loss['1extras2*'] = {'actlog0':np.float64, 'actlog1':np.float64}
            # metrics_loss['1extras2*'] = {'-actlog0':np.float64, '-actlog1':np.float64, '-actlog2':np.float64, '-actlog3':np.float64}
            # # metrics_loss['1extras1*'] = {'-ma':np.float64, '-ema':np.float64}
            # metrics_loss['1extras1*'] = {'-snr_loss':np.float64, '-std_loss':np.float64}
            # metrics_loss['1extras5'] = {'-snr_rtn':np.float64}
            # metrics_loss['1~extra3'] = {'-learn_rate':np.float64}
            # metrics_loss['1extra4'] = {'loss_meta':np.float64}
        if self.arch == 'AC':
            metrics_loss['1netsR'] = {'loss_action_lik':np.float64, 'loss_value_rep':np.float64}
            metrics_loss['1nets'] = {'loss_action':np.float64, 'loss_value':np.float64}
            # metrics_loss['1extras*'] = {'returns':np.float64, 'advantages':np.float64}
            metrics_loss['1extras2*'] = {'actlog0':np.float64, 'actlog1':np.float64}
        if self.arch == 'MU':
            metrics_loss['1~nets*'] = {'-loss_ma':np.float64, '-loss_action':np.float64}
            # metrics_loss['1extrasAR'] = {'loss_action_returns':np.float64}
            metrics_loss['1~extras'] = {'loss_trans':np.float64}
            metrics_loss['1~extras1'] = {'loss_value':np.float64}
            # metrics_loss['1~extras3'] = {'loss_gen':np.float64}
            # metrics_loss['1extras4'] = {'loss_act':np.float64}
            # metrics_loss['1extras2*'] = {'actlog0':np.float64, 'actlog1':np.float64}
            # metrics_loss['1extras1*'] = {'-snr_loss':np.float64, '-std_loss':np.float64}
            # metrics_loss['1~extra3'] = {'-learn_rate':np.float64}
            # metrics_loss['1~extra3'] = {'-lr_rep_action':np.float64}
            # metrics_loss['1~extra4'] = {'-lr_rep_trans':np.float64}
            # metrics_loss['1~extra5'] = {'-lr_action':np.float64}
            # metrics_loss['1~extra6'] = {'-lr_trans':np.float64}
        if self.trader:
            metrics_loss['2rewards*'] = {'equity_final=':np.float64, '-draw_total':np.float64}
            metrics_loss['1trader_sim_time'] = {'sim_time_secs':np.float64}
            # metrics_loss['1trader_draws'] = {'-drawdown_total':np.float64}

        for loss_group in metrics_loss.values():
            for k in loss_group.keys():
                if k.endswith('=') or k.endswith('+'): loss_group[k] = [0 for i in range(self.max_episodes)]
                else: loss_group[k] = [[] for i in range(self.max_episodes)]
        self.metrics_loss = metrics_loss

    def metrics_update(self, *args):
        args = list(args)
        # for i in range(1,len(args)): args[i] = args[i].item()
        log_metrics, episode, idx = args[0], args[1], 2
        for loss_group in self.metrics_loss.values():
            for k in loss_group.keys():
                if log_metrics[idx-2]:
                    if k.endswith('='): loss_group[k][episode] = args[idx]
                    elif k.endswith('+'): loss_group[k][episode] += args[idx]
                    else: loss_group[k][episode] += [args[idx]]
                idx += 1
        return np.asarray(0, np.int32) # dummy


    def env_reset(self, dummy):
        obs, info = self.env.reset(); reward, done = 0.0, False
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = gym_util.struc_to_feat(obs)
        else: rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        metrics = info['metrics'] if 'metrics' in info else [0]
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool), np.asarray([metrics], np.float64)]
        return rtn
    def env_step(self, *args): # args = tuple of ndarrays
        if hasattr(self.env,'np_struc'): action = gym_util.out_to_struc(list(args), self.env.action_dtype)
        else: action = gym_util.out_to_space(args, self.env.action_space, [0])
        obs, reward, terminated, truncated, info = self.env.step(action); done = (terminated or truncated)
        if self.env_render: self.env.render()
        if hasattr(self.env,'np_struc'): rtn = gym_util.struc_to_feat(obs)
        else: rtn = gym_util.space_to_feat(obs, self.env.observation_space)
        metrics = info['metrics'] if 'metrics' in info else [0]
        rtn += [np.asarray([[reward]], np.float64), np.asarray([[done]], bool), np.asarray([metrics], np.float64)]
        return rtn

    def check_stop(self, *args):
        # if keyboard.is_pressed('ctrl+alt+k'): return np.asarray(True, bool)
        if self.stop: self.stopped_episode = args[0].item(); return np.asarray(True, bool)
        if self.pause_episodes:
            print('PAUSED')
            while True:
                if keyboard.is_pressed('ctrl+alt+space'): break
                time.sleep(0.1)
        return np.asarray(False, bool)
    def on_stop(self):
        keyboard.unhook_all_hotkeys()
        print('STOPPING'); self.stop = True

    def checkpoints(self, *args):
        model_files = ""
        for net in self.layers:
            model_file = self.model_files[net.name]
            net.save_weights(model_file)
            model_files += ' '+model_file.split('/')[-1]
        print("SAVED{}".format(model_files))
        return np.asarray(0, np.int32) # dummy

    # TODO use ZMQ for remote messaging, latent pooling
    def transact_latents(self, *args):
        # args[0].shape
        return [np.asarray([0,1,2], np.float64), np.asarray([2,1,0], np.float64)]


    def reset_states(self, use_img=False):
        for net in self.layers:
            if hasattr(net, 'reset_states'): net.reset_states(use_img=use_img)



    def PG_actor(self, inputs, return_goal):
        print("tracing -> GeneralAI PG_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['step_shape'][1:])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['step_shape'][1:])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        # while step < self.max_steps and not inputs['dones'][-1][0]:
        while not inputs['dones'][-1][0]:
            # tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(inputs['obs'], [tf.TensorShape([None,None])]), (inputs['rewards'], tf.TensorShape([None,None])), (inputs['dones'], tf.TensorShape([None,None]))])
            # tf.autograph.experimental.set_loop_options(shape_invariants=[(outputs['rewards'], [None,1]), (outputs['dones'], [None,1]), (outputs['returns'], [None,1])])
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            action = [None]*self.action_spec_len
            # for i in range(self.action_spec_len):
            #     action[i] = tf.random.uniform((self.action_spec[i]['step_shape']), minval=self.action_spec[i]['min'], maxval=self.action_spec[i]['max'], dtype=self.action_spec[i]['dtype_out'])
            inputs_step = {'obs':inputs['obs'], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs['rewards']], 'return_goal':[return_goal]}
            # inputs_step = {'obs':[inputs['obs'][0]], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs['rewards']], 'return_goal':[return_goal]} # PG shkspr img tests
            # inputs_img = {'obs':[inputs['obs'][1]], 'step':[tf.reshape(step+1,(1,1))], 'reward_prev':[inputs['rewards']], 'return_goal':[return_goal]}
            # self.action.reset_states(use_img=True)
            # action_logits = self.action(inputs_step, use_img=True)
            # action_logits = self.action(inputs_img, use_img=True)
            # action_logits = self.action(inputs_step, use_img=True, store_real=True)
            action_logits = self.action(inputs_step)
            action_dist = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                # action_logits[i] = tf.constant(np.zeros(action_logits[i].shape),self.compute_dtype) # random actions for categorical
                action_dist[i] = self.action.dist[i](action_logits[i])
                action[i] = action_dist[i].sample()

            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][0])
                action_dis[i] = util.discretize(action[i][0], self.action_spec[i])

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            # inputs = {'obs':np_in[:-2], 'rewards':np_in[-2], 'dones':np_in[-1]}
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-3], np_in[-3], np_in[-2]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            # return_goal -= inputs['rewards']
            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    def PG_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI PG_learner_onestep")
        loss = {}
        loss_actions_lik = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_actlog = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(2,))

        inputs_rewards = tf.concat([self.rewards_zero, inputs['rewards']], axis=0)
        returns = inputs['returns'][0:1]; returns_calc = tf.squeeze(tf.cast(returns,self.compute_dtype)); returns_calc_orig = returns_calc # _loss-final
        avg_rtns, ma_rtns, ema_rtns, snr_rtns, std_rtns = util.stats_get(self.action.stats['rwd']); ema_rtns = tf.cast(ema_rtns,self.compute_dtype)
        returns_calc = returns_calc - ema_rtns # _rtns-ema # ma_rtns, avg_rtns
        # returns_calc = util.symlog(returns_calc) # _rtns-sym
        # returns_calc = returns_calc - util.symlog(ema_rtns) # _rtns-emaSL
        # returns_calc = snr_rtns # _rtns-snrO
        # returns_calc = returns_calc - (ema_rtns / (std_rtns+self.float_eps)) # _rtns-emaS
        # returns_calc = returns_calc * snr_rtns**3 # _rtns-snr3 # std_rtns
        # if returns_calc < 0.0: returns_calc = tf.constant(0,self.compute_dtype) # _rtns-emaC (needs _rtns-ema)
        # returns_calc = tf.math.abs(returns_calc - ema_rtns) # _rtns-emaA
        # returns_calc = returns_calc if returns_calc > self.float_eps else self.float_eps # _rtns-emaP
        # returns_calc = tf.constant(1,self.compute_dtype) if returns_calc > 0 else tf.constant(-1,self.compute_dtype) # _rtns-emaB

        # inputs['returns'] = util.symlog(inputs['returns']) # _rtnsSL
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            # returns = inputs['returns'][step:step+1]; returns_calc = tf.squeeze(tf.cast(returns,self.compute_dtype)); returns_calc_orig = returns_calc # _loss-finalN
            # returns_step = inputs['returns'][step:step+1]; returns_step_calc = tf.squeeze(tf.cast(returns_step,self.compute_dtype)) # _loss-rtnsI
            reward_calc = tf.squeeze(tf.cast(inputs['rewards'][step],self.compute_dtype))
            # if returns_calc < 30.0: returns_calc = tf.constant(0,self.compute_dtype) # _sparse
            # if returns_calc < 30.0: returns_calc = returns_calc / 30.0 # _clipL
            # if returns_calc > 170.0: returns_calc = (returns_calc - 170.0) / 30.0 + 170.0 # _clipH

            inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]}
            # inputs_step = {'obs':[obs[0]], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]} # PG shkspr img tests
            # inputs_img = {'obs':[obs[1]], 'step':[tf.reshape(step+1,(1,1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]}
            # self.action.reset_states(use_img=True)
            # action_logits = self.action(inputs_step, use_img=True)
            # action_logits = self.action(inputs_img, use_img=True)
            with tf.GradientTape() as tape_action:
                # action_logits = self.action(inputs_step, use_img=True, store_real=True)
                action_logits = self.action(inputs_step)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len):
                    # if self.action_spec[i]['dist_type'] == 'c': # _logit-norm
                    #     action_logits_norm = tf.norm(action_logits[i], ord=2, axis=-1, keepdims=True) + self.float_eps
                    #     action_logits[i] = tf.math.divide(action_logits[i], action_logits_norm) # / tf.constant(0.01,self.compute_dtype)
                    action_dist[i] = self.action.dist[i](action_logits[i])
                # loss_action = util.loss_PG(action_dist, action, returns)
                loss_action_lik = util.loss_likelihood(action_dist, action)
                # loss_action_lik = util.loss_likelihood(action_dist, obs); loss_action = loss_action_lik # _loss-direct
                # loss_action_lik = loss_action_lik - self.float_maxroot # _lSmr # causes NaN/inf
                # loss_action_lik = loss_action_lik - self.float_eps_max # _lSem
                # loss_action_lik = loss_action_lik - self.loss_scale # _lSls
                loss_action = loss_action_lik * returns_calc
                # loss_action = loss_action_lik - returns_calc # _rtnsS
                # loss_action = loss_action_lik * returns_calc - returns_calc # _rtnsMS
                # loss_action = loss_action_lik * (returns_calc + reward_calc) # _rtnsR
                # loss_action = loss_action_lik * reward_calc # _loss-rwd
                # loss_action = loss_action_lik # _loss-udRL
                # loss_action = loss_action_lik * (returns_calc + 1) # _rtnsP1
                # loss_action = loss_action * returns_step_calc # _loss-rtnsI
                # loss_action = loss_action * reward_calc # _loss-rwdO
                loss_action = loss_action + loss_action_lik * reward_calc # _loss-rwdG
                # loss_action = loss_action - returns_calc # _loss-rtns # no gradients
                # loss_action = loss_action - reward_calc # _loss-rwdS # no gradients
                # loss_action = loss_action - ema_rtns # _rtnsEM _loss-rtnsS # no gradients
                # loss_action = loss_action + util.loss_entropy(action_dist, 1e-0) # , 1e-3 # _rtnsE _loss-ent
                for i in range(self.action_spec_len): loss_action = loss_action + action_dist[i].params_loss(action_logits[i]) # * tf.math.abs(returns_calc + reward_calc) # _loss-logits
                # loss_action = self.action.optimizer['action'].get_scaled_loss(loss_action)
                # loss_action = loss_action * self.loss_scale # _loss-scale
            if loss_action_lik > self.float_eps: # _grad-lim-eps
            # if reward_calc > tf.constant(0,self.compute_dtype): # _grad-lim-rwd
            # if tf.math.abs(reward_calc) > self.float_eps and loss_action_lik > self.float_eps:
                gradients = tape_action.gradient(loss_action, self.action.trainable_variables)
                # gradients = self.action.optimizer['action'].get_unscaled_gradients(gradients)
                # for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale # _loss-scale
                self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
            loss_actions_lik = loss_actions_lik.write(step, loss_action_lik / self.action_total_size)
            loss_actions = loss_actions.write(step, loss_action) # / self.loss_scale
            metric_actlog = metric_actlog.write(step, action_logits[0][0][0:2])

        loss['action_lik'], loss['action'], loss['actlog'] = loss_actions_lik.concat(), loss_actions.concat(), metric_actlog.stack()
        return loss

    def PG(self):
        print("tracing -> GeneralAI PG"); tf.print("RUNNING")
        # loss_meta, ma_loss_lowest = tf.constant([0],self.compute_dtype), self.float_maxroot
        # return_goal = tf.constant([[200]],tf.float64) # _rpC
        return_goal = tf.constant([[-self.loss_scale.numpy()]],tf.float64) # _rpB
        episode, stop = tf.constant(0), tf.constant(False)
        while episode < self.max_episodes and not stop:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1)
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-3], 'rewards':np_in[-3], 'dones':np_in[-2]}; env_metrics = np_in[-1][0]

            # TODO how unlimited length episodes without sacrificing returns signal?
            self.reset_states(); outputs, inputs = self.PG_actor(inputs, return_goal)
            # util.stats_update(self.action.stats['rwd'], tf.math.reduce_sum(outputs['rewards'])); avg, ma, ema, snr, std = util.stats_get(self.action.stats['rwd'])
            rewards_total = outputs['returns'][0][0] # tf.math.reduce_sum(outputs['rewards'])
            util.stats_update(self.action.stats['rwd'], rewards_total); avg, ma, ema, snr, std = util.stats_get(self.action.stats['rwd'])
            # return_goal = tf.reshape((ma + 10.0),(1,1)) # _rpP
            # if outputs['returns'][0:1] > return_goal: return_goal = tf.reshape(outputs['returns'][0:1],(1,1)); tf.print(return_goal) # _rpB

            # # meta learn the optimizer learn rate / step size
            # _, _, _, _, std = util.stats_get(self.action.stats['loss'])
            # obs = [self.action.stats['loss']['iter'].value(), tf.cast(ma,self.compute_dtype), std]
            # inputs_meta = {'obs':[tf.expand_dims(tf.stack(obs,0),0)]}

            # learn_rate = self.action.optimizer['action'].learning_rate
            # with tf.GradientTape() as tape_meta:
            #     meta_logits = self.meta(inputs_meta); meta_dist = self.meta.dist[0](meta_logits[0])
            #     loss_meta = util.loss_PG(meta_dist, tf.reshape(learn_rate,(1,1)), tf.reshape(rewards_total,(1,1)))
            # gradients = tape_meta.gradient(loss_meta, self.meta.trainable_variables)
            # self.meta.optimizer['meta'].apply_gradients(zip(gradients, self.meta.trainable_variables))

            # meta_logits = self.meta(inputs_meta); meta_dist = self.meta.dist[0](meta_logits[0])
            # learn_rate = meta_dist.sample()
            # self.action.optimizer['action'].learning_rate = util.discretize(learn_rate, self.meta_spec[0])
            # # self.action.optimizer['action'].learning_rate = tf.squeeze(tf.cast(learn_rate, tf.float64))


            self.reset_states(); loss = self.PG_learner_onestep(outputs)
            util.stats_update(self.action.stats['loss'], tf.math.reduce_mean(loss['action_lik'])); avg_loss, ma_loss, ema_loss, snr_loss, std_loss = util.stats_get(self.action.stats['loss'])

            # self.action.optimizer['action'].learning_rate = self.action_get_learn_rate(ma_loss) # _lr-loss
            # self.action.optimizer['action'].learning_rate = self.action_get_learn_rate(std) # _lr-rwd-std
            # self.action.optimizer['action'].learning_rate = tf.math.exp(episode / self.max_episodes * (-15.0 + 9.7) - 9.7) # _lr-scale
            self.action.optimizer['action'].learning_rate = self.learn_rates['action'] * snr_loss**np.e # **3 # _lr-snr3
            # self.action.optimizer['action'].learning_rate = self.learn_rates['action'] * (1.0 - rewards_total / 200.0) + self.float_eps # _lr-rwd-lin-scale

            # if ma_loss < ma_loss_lowest: ma_loss_lowest = ma_loss
            # # if self.action.stats['loss']['iter'] > 10 and std_loss < 1.0 and tf.math.abs(ma_loss) < 1.0:
            # if snr_loss < 0.5 and std_loss < 0.2 and tf.math.abs(ma_loss) < 0.1:
            # if self.action.stats['loss']['iter'] > 16 and tf.math.abs(ma_loss) < 1e-1: # self.float_eps 1e-1 # _rst
            #     tf.print("net_reset (action) at:", episode, " lr:", self.action.optimizer['action'].learning_rate, " ma_loss:", ma_loss, " snr_loss:", snr_loss, " std_loss:", std_loss)
            #     util.net_reset(self.action); self.action.optimizer['action'].learning_rate = self.learn_rates['action']
            #     # self.action.optimizer['action'].learning_rate = tf.random.uniform((), dtype=tf.float64, maxval=self.learn_rates['action'], minval=self.float64_eps) # _lr-rnd-linear
            #     # self.action.optimizer['action'].learning_rate = tf.math.exp(tf.random.uniform((), dtype=tf.float64, maxval=-7, minval=-16)) # _lr-rnd-exp

            log_metrics = [True,True,True,True,True,True,True,True,True,True,True,True,True,True]
            metrics = [log_metrics, episode, ema, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                ma_loss, tf.math.reduce_mean(loss['action_lik']), # tf.math.reduce_mean(outputs['returns']),
                tf.math.reduce_mean(loss['action']),
                tf.math.reduce_mean(loss['actlog'][:,0]), tf.math.reduce_mean(loss['actlog'][:,1]),
                # tf.math.reduce_mean(loss['actlog'][:,2]), tf.math.reduce_mean(loss['actlog'][:,3]),
                # snr,
                # self.action.optimizer['action'].learning_rate,
                # loss_meta[0],
            ]
            if self.trader:
                del metrics[2]; metrics[2], metrics[3] = inputs['obs'][4][-1][0], env_metrics[0]
                metrics += [inputs['obs'][0][-1][0] - outputs['obs'][0][0][0], env_metrics[1]]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

            if self.save_model:
                if episode > tf.constant(0) and episode % self.chkpts == tf.constant(0): tf.numpy_function(self.checkpoints, [tf.constant(0)], [tf.int32])
            stop = tf.numpy_function(self.check_stop, [episode], tf.bool); stop.set_shape(())
            episode += 1
        # tf.print("ma_loss_lowest", ma_loss_lowest)



    def AC_actor(self, inputs, return_goal):
        print("tracing -> GeneralAI AC_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['step_shape'][1:])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['step_shape'][1:])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        step = tf.constant(0)
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            inputs_step = {'obs':inputs['obs'], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs['rewards']], 'return_goal':[return_goal]}
            rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
            latent_rep = rep_dist.sample()

            action = [None]*self.action_spec_len
            action_logits = self.action(latent_rep)
            for i in range(self.action_spec_len):
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()

            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][0])
                action_dis[i] = util.discretize(action[i][0], self.action_spec[i])

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-3], np_in[-3], np_in[-2]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            # return_goal -= inputs['rewards']
            step += 1

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        return outputs, inputs

    def AC_rep_learner(self, inputs, training=True):
        print("tracing -> GeneralAI AC_rep_learner")
        loss = {}
        loss_values = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_rewards = tf.concat([self.rewards_zero, inputs['rewards']], axis=0)
        returns = inputs['returns'][0:1] # _loss-final
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            # returns = inputs['returns'][step:step+1]
            returns_calc = tf.squeeze(tf.cast(returns,self.compute_dtype),axis=-1)
            reward_calc = tf.cast(inputs['rewards'][step],self.compute_dtype)

            inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]}
            with tf.GradientTape(persistent=True) as tape_value, tf.GradientTape(persistent=True) as tape_action:
                rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
                latent_rep = rep_dist.sample()

            inputs_value = {'obs':latent_rep, 'actions':action}
            with tape_value:
                value_logits = self.value(inputs_value); value_dist = self.value.dist[0](value_logits[0])
                values = value_dist.sample()
                if self.value_cont: loss_value = util.loss_likelihood(value_dist, returns)
                else: loss_value = util.loss_diff(values, returns)
            gradients = tape_value.gradient(loss_value, self.rep.trainable_variables)
            self.rep.optimizer['value'].apply_gradients(zip(gradients, self.rep.trainable_variables))
            loss_values = loss_values.write(step, loss_value)

            with tape_action:
                action_logits = self.action(latent_rep)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                loss_action_lik = util.loss_likelihood(action_dist, action)
                loss_action_lik = loss_action_lik - self.loss_scale # _lSls
                loss_action = loss_action_lik * (returns_calc + loss_value) # _lEp5 *
                loss_action = loss_action * self.loss_scale
            gradients = tape_action.gradient(loss_action, self.rep.trainable_variables)
            # gradients = tape_action.gradient(loss_action_lik, self.rep.trainable_variables) # _rep-lik
            for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
            self.rep.optimizer['action'].apply_gradients(zip(gradients, self.rep.trainable_variables))
            loss_actions = loss_actions.write(step, loss_action_lik)

        loss['value'], loss['action'] = loss_values.concat(), loss_actions.concat()
        return loss

    def AC_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI AC_learner_onestep")
        loss = {}
        loss_values = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions_lik = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_actlog = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(2,))
        # metric_advantages = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        # return_goal = tf.constant([[200.0]], tf.float64)
        inputs_rewards = tf.concat([self.rewards_zero, inputs['rewards']], axis=0)
        returns = inputs['returns'][0:1] # _loss-final
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])
            # returns = inputs['returns'][step:step+1]
            returns_calc = tf.squeeze(tf.cast(returns,self.compute_dtype),axis=-1)
            reward_calc = tf.cast(inputs['rewards'][step],self.compute_dtype)

            inputs_step = {'obs':obs, 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs_rewards[step:step+1]], 'return_goal':[returns]}
            rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
            latent_rep = rep_dist.sample()

            inputs_value = {'obs':latent_rep, 'actions':action}
            with tf.GradientTape() as tape_value:
                value_logits = self.value(inputs_value); value_dist = self.value.dist[0](value_logits[0])
                values = value_dist.sample()
                if self.value_cont: loss_value = util.loss_likelihood(value_dist, returns)
                else: loss_value = util.loss_diff(values, returns)
            gradients = tape_value.gradient(loss_value, self.value.trainable_variables)
            self.value.optimizer['value'].apply_gradients(zip(gradients, self.value.trainable_variables))
            loss_values = loss_values.write(step, loss_value)

            with tf.GradientTape() as tape_action:
                action_logits = self.action(latent_rep)
                action_dist = [None]*self.action_spec_len
                for i in range(self.action_spec_len): action_dist[i] = self.action.dist[i](action_logits[i])
                # loss_action = util.loss_PG(action_dist, action, returns, values)
                # loss_action = util.loss_PG(action_dist, action, returns, values, returns_target=return_goal) # _lPGt
                # loss_action = util.loss_PG(action_dist, action, loss_value) # _lPGv
                loss_action_lik = util.loss_likelihood(action_dist, action)
                loss_action_lik = loss_action_lik - self.loss_scale # _lSls
                # loss_action = loss_action_lik * returns_calc # _lEpA
                # loss_action = loss_action_lik * returns_calc - tf.squeeze(values,axis=-1))
                # loss_action = loss_action_lik * tf.math.exp(-loss_value) # _lEp1
                # loss_action = loss_action_lik * (1.0 - tf.math.exp(-loss_value)) # _lEpC
                # loss_action = loss_action_lik * (-loss_value) # _lEp2
                # loss_action = loss_action_lik * loss_value # _lEp9 *
                # loss_action = loss_action_lik * ((tf.math.exp(-loss_value) + 1.0) * 100.0) # _lEp3
                # loss_action = loss_action_lik * (returns_calc - loss_value) # _lEp4
                loss_action = loss_action_lik * (returns_calc + loss_value) # _lEp5 *
                # loss_action = loss_action_lik * (tf.math.exp(-loss_value) + 1.0) # _lEp6
                # loss_action = loss_action_lik * ((returns_calc / 200.0) - tf.math.exp(-loss_value)) # _lEp7
                # loss_action = loss_action_lik * ((returns_calc / 200.0) - tf.math.exp(-loss_value) + 1.0) / 2.0 # _lEp8
                # loss_action = loss_action_lik * (returns_calc + loss_value + reward_calc) # _lEp5R *
                loss_action = loss_action * self.loss_scale
            gradients = tape_action.gradient(loss_action, self.action.trainable_variables)
            for i in range(len(gradients)): gradients[i] = gradients[i] / self.loss_scale
            self.action.optimizer['action'].apply_gradients(zip(gradients, self.action.trainable_variables))
            loss_actions_lik = loss_actions_lik.write(step, loss_action_lik / self.action_total_size)
            loss_actions = loss_actions.write(step, loss_action / self.loss_scale)
            # metric_advantages = metric_advantages.write(step, (returns - tf.cast(values,tf.float64))[0])
            metric_actlog = metric_actlog.write(step, action_logits[0][0][0:2])
            # return_goal -= inputs['rewards'][step:step+1]; return_goal.set_shape((1,1))

        loss['value'], loss['action_lik'], loss['action'], loss['actlog'] = loss_values.concat(), loss_actions_lik.concat(), loss_actions.concat(), metric_actlog.stack()
        # loss['advantages'] = metric_advantages.concat()
        return loss

    def AC(self):
        print("tracing -> GeneralAI AC"); tf.print("RUNNING")
        return_goal, ma = tf.constant([[-self.loss_scale.numpy()]], tf.float64), tf.constant(0,tf.float64)
        episode, stop = tf.constant(0), tf.constant(False)
        while episode < self.max_episodes and not stop:
            tf.autograph.experimental.set_loop_options(parallel_iterations=1) # TODO parallel wont work with single instance env, will this work multiple?
            np_in = tf.numpy_function(self.env_reset, [tf.constant(0)], self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs = {'obs':np_in[:-3], 'rewards':np_in[-3], 'dones':np_in[-2]}; env_metrics = np_in[-1][0]

            self.reset_states(); outputs, inputs = self.AC_actor(inputs, return_goal)
            rewards_total = outputs['returns'][0][0] # tf.math.reduce_sum(outputs['rewards'])
            util.stats_update(self.action.stats['rwd'], rewards_total); avg, ma, ema, snr, std = util.stats_get(self.action.stats['rwd'])
            self.reset_states(); loss_rep = self.AC_rep_learner(outputs)
            self.reset_states(); loss = self.AC_learner_onestep(outputs)

            # return_goal = tf.constant([[200.0]], tf.float64)
            # return_goal = tf.reshape((ma + 10.0),(1,1)) # _rpP
            if outputs['returns'][0:1] > return_goal: return_goal = tf.reshape(outputs['returns'][0:1],(1,1)); tf.print(return_goal) # _rpB

            log_metrics = [True,True,True,True,True,True,True,True,True,True,True,True,True,True]
            metrics = [log_metrics, episode, ma, tf.math.reduce_sum(outputs['rewards']), outputs['rewards'][-1][0], tf.shape(outputs['rewards'])[0],
                tf.math.reduce_mean(loss['action_lik']), tf.math.reduce_mean(loss_rep['value']),
                tf.math.reduce_mean(loss['action']), tf.math.reduce_mean(loss['value']),
                # tf.math.reduce_mean(outputs['returns']), tf.math.reduce_mean(loss['advantages']),
                tf.math.reduce_mean(loss['actlog'][:,0]), tf.math.reduce_mean(loss['actlog'][:,1]),
            ]
            if self.trader:
                del metrics[2]; metrics[2], metrics[3] = inputs['obs'][4][-1][0], env_metrics[0]
                metrics += [inputs['obs'][0][-1][0] - outputs['obs'][0][0][0], env_metrics[1]]
            dummy = tf.numpy_function(self.metrics_update, metrics, [tf.int32])

            if self.save_model:
                if episode > tf.constant(0) and episode % self.chkpts == tf.constant(0): tf.numpy_function(self.checkpoints, [tf.constant(0)], [tf.int32])
            stop = tf.numpy_function(self.check_stop, [episode], tf.bool); stop.set_shape(())
            episode += 1



    def MU_actor(self, inputs, return_goal):
        print("tracing -> GeneralAI MU_actor")
        obs, actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): obs[i] = tf.TensorArray(self.obs_spec[i]['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.obs_spec[i]['step_shape'][1:])
        for i in range(self.action_spec_len): actions[i] = tf.TensorArray(self.action_spec[i]['dtype_out'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.action_spec[i]['step_shape'][1:])
        rewards = tf.TensorArray(tf.float64, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        dones = tf.TensorArray(tf.bool, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        returns = tf.TensorArray(tf.float64, size=0, dynamic_size=True, infer_shape=False, element_shape=(1,))
        latents_rep = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])
        # latents_trans = tf.TensorArray(self.latent_spec['dtype'], size=1, dynamic_size=True, infer_shape=False, element_shape=self.latent_spec['step_shape'])

        action_prev = self.action_zero_out
        step = tf.constant(0)
        while not inputs['dones'][-1][0]:
            for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])

            inputs_step = {'obs':inputs['obs'], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs['rewards']], 'done_prev':[inputs['dones']], 'actions':action_prev, 'return_goal':[return_goal]}
            rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
            latent_rep = rep_dist.sample()
            latents_rep = latents_rep.write(step, latent_rep)

            # TODO train trans for reconstruct and next state, condition action with reconstruct loss
            trans_logits = self.trans(latent_rep); trans_dist = self.trans.dist[0](trans_logits[0])
            latent_trans = trans_dist.sample()
            # latents_trans = latents_trans.write(step, latent_trans[0])

            self.action.net.layer_attn[0]._memory_img[-self.mem_img_size:].assign(latent_trans)
            self.action.net.layer_attn[0]._mem_idx_img.assign(self.max_steps - self.mem_img_size)
            action_logits = self.action(latent_rep, store_memory=False, use_img=True, store_real=True) # _act-hist
            # action_logits = self.action(latent_trans)
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                # action_logits[i] = tf.constant(np.zeros(action_logits[i].shape),self.compute_dtype) # random actions for categorical
                action_dist = self.action.dist[i](action_logits[i])
                action[i] = action_dist.sample()
            action_prev = action

            action_dis = [None]*self.action_spec_len
            for i in range(self.action_spec_len):
                actions[i] = actions[i].write(step, action[i][0])
                action_dis[i] = util.discretize(action[i][0], self.action_spec[i])

            np_in = tf.numpy_function(self.env_step, action_dis, self.gym_step_dtypes)
            for i in range(len(np_in)): np_in[i].set_shape(self.gym_step_shapes[i])
            inputs['obs'], inputs['rewards'], inputs['dones'] = np_in[:-3], np_in[-3], np_in[-2]

            rewards = rewards.write(step, inputs['rewards'][-1])
            dones = dones.write(step, inputs['dones'][-1])
            returns = returns.write(step, [self.float64_zero])
            returns_updt = returns.stack()
            returns_updt = returns_updt + inputs['rewards'][-1]
            returns = returns.unstack(returns_updt)

            step += 1
        for i in range(self.obs_spec_len): obs[i] = obs[i].write(step, inputs['obs'][i][-1])
        inputs_step = {'obs':inputs['obs'], 'step':[tf.reshape(step,(1,1))], 'reward_prev':[inputs['rewards']], 'done_prev':[inputs['dones']], 'actions':action_prev, 'return_goal':[return_goal]}
        rep_logits = self.rep(inputs_step); rep_dist = self.rep.dist(rep_logits)
        latent_rep = rep_dist.sample()
        latents_rep = latents_rep.write(step, latent_rep)

        outputs = {}
        out_obs, out_actions = [None]*self.obs_spec_len, [None]*self.action_spec_len
        for i in range(self.obs_spec_len): out_obs[i] = obs[i].stack()
        for i in range(self.action_spec_len): out_actions[i] = actions[i].stack()
        outputs['obs'], outputs['actions'], outputs['rewards'], outputs['dones'], outputs['returns'] = out_obs, out_actions, rewards.stack(), dones.stack(), returns.stack()
        outputs['latents_rep'] = latents_rep.stack()
        # outputs['latents_trans'] = latents_trans.stack()
        return outputs, inputs

    def MU_learner_onestep(self, inputs, training=True):
        print("tracing -> GeneralAI MU_learner_onestep")
        loss = {}
        loss_actions_lik = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_actions = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_transs = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_values = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        loss_gens = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))
        metric_actlog = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(2,))
        loss_acts = tf.TensorArray(self.compute_dtype, size=1, dynamic_size=True, infer_shape=False, element_shape=(1,))

        inputs_rewards, inputs_dones, inputs_latents_rep, action_prev = tf.concat([self.rewards_zero, inputs['rewards']], axis=0), tf.concat([self.dones_zero, inputs['dones']], axis=0), tf.squeeze(inputs['latents_rep'], axis=1), self.action_zero_out
        returns = inputs['returns'][0:1]; returns_calc = tf.squeeze(tf.cast(returns,self.compute_dtype)); returns_calc_orig = returns_calc # _loss-final
        avg_rtns, ma_rtns, ema_rtns, snr_rtns, std_rtns = util.stats_get(self.action.stats['rwd']); ema_rtns = tf.cast(ema_rtns,self.compute_dtype)
        returns_calc = returns_calc - ema_rtns # _rtns-ema
        # returns_calc = util.symlog(returns_calc) # _rtns-sym

        num_reps = tf.shape(inputs['latents_rep'])[0]
        for step in tf.range(tf.shape(inputs['dones'])[0]):
            obs = [None]*self.obs_spec_len
            for i in range(self.obs_spec_len): obs[i] = inputs['obs'][i][step:step+1]; obs[i].set_shape(self.obs_spec[i]['step_shape'])
            action = [None]*self.action_spec_len
            for i in range(self.action_spec_len): action[i] = inputs['actions'][i][step:step+1]; action[i].set_shape(self.action_spec[i]['step_shape'])