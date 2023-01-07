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

        self._env_clock, self._env_speed, self._env_render, self._env_np_struc = env_clock, env_speed, env_render, np_struc
        reward_done_zero = [np.frombuffer(np.asarray(0, np.float64), dtype=np.uint8), np.frombuffer(np.asarray(False, bool), dtype=np.uint8)]
        self._reward_done_zero = reward_done_zero

        self._action_timing, self._obs_timing = False, False
        if not (isinstance(env.action_space, gym.spaces.Dict) and 'timedelta' in env.action_space.spaces):
            self._action_timing = True
            self.action_space = gym.spaces.Dict()
            self.action_space.spaces['timedelta'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            self.action_space.spaces['origspace'] = env.action_space
            if np_struc: self.action_dtype = np.dtype([('timedelta', 'f8'), ('origspace', env.action_dtype)])
        if not (isinstance(env.observation_space, gym.spaces.Dict) and 'timestamp' in env.observation_space.spaces):
            self._obs_timing = True
            self.observation_space = gym.spaces.Dict()
    