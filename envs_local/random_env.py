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
        # obs_smpl = sel