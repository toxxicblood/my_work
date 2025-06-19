import gym
from gym import spaces
import numpy as np
import pandas as pd

class ForexEnv (gym.Env):
    def __init__(self, df, window_size=16):
        super().__init__()#(ForexEnv, self).__init__()
        self.df = df.reset_index(drop=True) #df
        self.window = window_size
        self.reset()
        # Define action and observation space
        self.action_space = spaces.Discrete(3) # long, short , hold
        #self.observation_space = spaces.Box(low=-1, high=1, shape=(self.window, 5), dtype=np.float32)
        n_features = df.shape[1]
        self.observation_space = spaces.Box(-np.inf, np.inf, (window_size, n_features), dtype=np.float32)


    def reset(self):
        self.idx = self.window
        self.position = 0 # -1: short, 0: neutral, 1: long
        self.total_profit = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return self.df.iloc[self.idx-self.window:self.idx].values.astype(np.float32)

    def step1(self, action):
        if self.idx >= len(self.df):
            return self._get_observation(), 0, True, {}
        price_diff = (self.df.Close.iloc[self.idx] - self.df.Close.iloc[self.idx-1]) / self.df.Close.iloc[self.idx-1]
        reward = {0: -price_diff, 1: price_diff, 2: 0}[action]
        self.idx += 1
        done = self.idx >= len(self.df)
        return self._get_observation(), reward, done, {}

    def step2(self, action):
        # Compute reward
        reward = 0
        pct_change = (self.df[self.idx[0]])
        if action == 1:#long
            reward = pct_change
        elif action == 2: #short
            reward = -pct_change
        else: #hold
            reward = 0

        self.total_profit += reward
        self.position = action
        self.idx += 1
        self.done = self.idx >= len(self.df)
        return self._get_observation(), reward, self.done, {}
    def render(self, mode='human'):
        print(f"Step: {self.idx}, Position: {self.position}, Total Profit: {self.total_profit}")


    def close(self):
        pass
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    #normalise the reward function as described
    #make sure env supports long, short and neutral
