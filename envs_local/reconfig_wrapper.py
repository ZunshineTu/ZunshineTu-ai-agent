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
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self.reward_range = env.reward_range
        self.spec = env.spec

        self.reconfig_act = False
        if isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape[-1] > 1:
            self.reconfig_act = True
            num_feat_act = env.action_space.shape[-1]; self.num_feat_act = num_feat_act
            act_shape = env.action_space.shape[:-1] + (1,)
            act = []
            for i in range(num_feat_act):
                feat = gym.spaces.Box(
                    low=env.action_space.low[...,i:i+1],
                    high=env.action_space.high[...,i:i+1],
                    shape=act_shape,
                    dtype=env.action_space.dtype
        