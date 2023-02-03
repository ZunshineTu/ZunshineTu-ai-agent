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
            self.observation_space.spaces['timestamp'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
            self.observation_space.spaces['origspace'] = env.observation_space
            if np_struc: self.obs_dtype = np.dtype([('timestamp', 'f8'), ('origspace', env.obs_dtype)])

        if np_struc:
            reward_size, done_size = np.dtype(np.float64).itemsize, np.dtype(bool).itemsize
            self._obs_idx, self._done_idx = -(reward_size + done_size), -done_size
            action_size, obs_size = env.action_dtype.itemsize, self.obs_dtype.itemsize + reward_size + done_size
        else:
            idx = 0; action_idxs = gym_util.space_to_bytes(env.action_space.sample(), env.action_space)
            for i in range(len(action_idxs)): idx += action_idxs[i].size; action_idxs[i] = idx
            action_idxs = [0] + action_idxs

            idx = 0; obs_idxs = gym_util.space_to_bytes(self.observation_space.sample(), self.observation_space)
            obs_idxs += reward_done_zero
            for i in range(len(obs_idxs)): idx += obs_idxs[i].size; obs_idxs[i] = idx
            obs_idxs = [0] + obs_idxs

            self._action_idxs, self._obs_idxs = action_idxs, obs_idxs
            action_size, obs_size = action_idxs[-1], obs_idxs[-1]

        # self._lock_print = mp.Lock()
        self._proc_ctrl = mp.Value('b', 0) # 1 = close, 0 = reset, -1 = step, -2 = done
        self._action_shared = mp.sharedctypes.Array('B', act