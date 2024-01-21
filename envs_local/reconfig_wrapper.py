# from collections import OrderedDict
# import copy
import numpy as np
np.set_printoptions(precision=8, suppress=True, linewidth=400, threshold=100)
# np.random.seed(0)
import gymnasium as gym


class ReconfigWrapperEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, env):
        super(ReconfigWrapperEnv, self).__init__()
        self.env = env
        self.action_space =