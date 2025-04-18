import gym
from gym import spaces
import numpy as np
import pandas as pd

class ForexEnv (gym.Env):
    def init__(self, df, window_size=16):
        super().__init__()#(ForexEnv, self).__init__()
        self.df = df.reset_index(drop=True) #df
        self.window = window_size
        self.action_space = spaces.Discrete(3) # long, short , hold
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(self.window, 5), dtype=np.float32)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, (window_size, n_features), dtype=np.float32)
        self.reset()

    def reset(self):
        self.idx = self.window
        self.position = 0
        return self._get_obs()

    def _get_obs(self):
        return self.df.iloc[self.idx-self.window]

    def step(self, action):
        #compute reward
        #price_diff = self.df.iloc[self.idx]['Close'] - self.df.iloc[self.idx-1]['Close']
        price_diff = (self.df.Close.iloc[self.idx] - self.df.Close.iloc[self.idx-1]) / self.df.Close.iloc[self.idx-1]
        reward = {0: -price_diff, 1: price_diff, 2: 0}[action] # long, short , hold
        self.idx += 1
        done = self.idx >= len(self.df)
        return self._get_obs(), reward, done, {}