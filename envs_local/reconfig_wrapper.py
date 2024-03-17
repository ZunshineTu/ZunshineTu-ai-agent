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
                )
                act.append(feat)
            act = gym.spaces.Tuple(act)
            self.action_space = act

        self.reconfig_obs = False
        if isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.shape[-1] > 1:
            self.reconfig_obs = True
            num_feat_obs = env.observation_space.shape[-1]; self.num_feat_obs = num_feat_obs
            obs_shape = env.observation_space.shape[:-1] + (1,)
            obs = []
            for i in range(num_feat_obs):
                feat = gym.spaces.Box(
                    low=env.observation_space.low[...,i:i+1],
                    high=env.observation_space.high[...,i:i+1],
                    shape=obs_shape,
                    dtype=env.observation_space.dtype
                )
                obs.append(feat)
            obs.append(env.observation_space)
        