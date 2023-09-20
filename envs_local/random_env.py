import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gymnasium as gym
import gym_util

# TODO auto make the dtype struc from space for numpy dtype compatability with gym, need to include space it has more info like low,high
# def gym_space_to_dtype(space):
#     pass

class RandomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env_np_struc):
        super(RandomEnv, self).__init__()
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self.reward_range = (-np.inf,+np.inf)
        # self.obs_zero = gym.spaces.flatten(self.observation_space, self.observation_space.sample())
        # self.action_spec, self.action_zero, self.action_zero_out = gym_util.get_spec(self.action_space)
        # self.obs_spec, self.obs_zero, self.obs_zero_out = gym_util.get_spec(self.observation_space)
        # obs_dtype = gym_space_to_dtype(self.observation_space)
        # action_smpl = self.action_space.sample()
        # obs_smpl = self.observation_space.sample()

        if env_np_struc: self.np_struc = True
        if env_np_struc:
            action_dtype = self._action_space_struc()
            obs_dtype = self._obs_space_struc()
            self.action_dtype, self.obs_dtype = action_dtype, obs_dtype
            action_zero = np.zeros((1,), self.action_dtype)
            obs_zero = np.zeros((1,), self.obs_dtype)
        else:
            action_zero = gym_util.get_space_zero(self.action_space)
            obs_zero = gym_util.get_space_zero(self.observation_space)
        self.action_zero, self.obs_zero = action_zero, obs_zero

        self.state = self.action_zero, self.obs_zero, np.float64(0.0), False, {}

    def step(self, action):
        return self._request(action)
    def reset(self):
        return self._request(None)[0], {}
    def render(self, mode='human', close=False):
        action, obs, reward, done, info = self.state
        if action is None: print("{}\n".format(obs))
        else: print("{}\t\t--> {:.18f}{}\n{}\n".format(action, reward, (' DONE!' if done else ''), obs))



    def _action_space(self):
        # return gym.spaces.Dict({ # sorted by name
        #     'six': gym.spaces.Discrete(6), # int
        #     'bin': gym.spaces.MultiBinary(6), # np.ndarray([1, 0, 1, 0, 0, 1], dtype=int8)
        #     'mul': gym.spaces.MultiDiscrete([6,2]), # np.ndarray([3, 0], dtype=int64)
        #     'val': gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.float64)} # np.ndarray(2.22744361)
        # )
        # return gym.spaces.Discrete(6)
        # return gym.spaces.MultiDiscrete([6,2])
        # return gym.spaces.MultiBinary(6)
        # return gym.spaces.Tuple((gym.spaces.Discrete(6), gym.spaces.MultiBinary(6)))
        # return gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)
        # return gym.spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float64)

        # action_space = gym.spaces.Tuple([])

        # action_space.spaces.append(gym.spaces.Discrete(4))

        # action_space_sub = gym.spaces.Tuple([])
        # action_space_sub.spaces.append(gym.spaces.Discrete(8))
        # action_space_sub.spaces.append(gym.spaces.Box(low=0, high=255, shape=(3,2), dtype=np.uint8))
        # action_space.spaces.append(action_space_sub)

        # action_space.spaces.append(gym.spaces.Discrete(6))

        # action_space_sub2 = gym.spaces.Dict()
        # action_space_sub2.spaces['test'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(2,), dtype=np.float64)
        # action_space.spaces.append(action_space_sub2)

        action_space = gym.spaces.Dict()

        action_space.spaces['dt_sub'] = gym.spaces.Dict()
        action_space.spaces['dt_sub'].spaces['float64'] = gym.spaces.Box(low=np.NINF, high=np.inf, shape=(2,), dtype=np.float64)
        action_space.spaces['dt_sub'].spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(2,), dtype=np.uint8)

        action_space.spaces['byte'] = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
        action_space.spaces['2Darray'] = gym.spaces.Box(low=0, high=255, shape=(2,3), dtype=np.uint8)
        action_space.spaces['3Darray'] = gym.spaces.Box(low=0, high=255, shape=(3,3,3), dtype=np.int32)
        action_space.spaces['discrete6'] = gym.spaces.Discrete(6)
        # action_space.spaces['multidiscrete2x6'] = gym.spaces.MultiDiscrete([6,6]) # TODO
        action_space.spaces['float64'] = gym.spaces.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float64)
        action_space.spaces['bools'] = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=bool)


        return action_space

    def _action_space_struc(self):
        dt_sub = np.dtype([
            ('float64', np.float64, (2,)),
            ('byte', np.uint8, (2,)),
        ])
        dtype = np.dtype([
            ('dt_sub', dt_sub),
            ('byte', np.uint8),
            ('2Darray', np.uint8, (2,3)),
            ('3Darray', np.int32, (3,3,3)),
            ('discrete6', np.int64),
            ('float64', np.float64),
 