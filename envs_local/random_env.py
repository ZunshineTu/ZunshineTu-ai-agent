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
        self.reward_range = (-