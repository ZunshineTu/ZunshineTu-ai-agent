from collections import OrderedDict
import time
import multiprocessing as mp
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gymnasium as gym
import gym_util


class AsyncWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env, env_clock, env_speed, env_render):
        super(AsyncWrapperEnv, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        np_struc = hasattr(env,'np_struc')
        if np_struc: self.np_struc, self.action_dtype, self.obs_dtype = env.np_struc, env.action_dtype, env.obs_dtype

        self._env_clock, self._e