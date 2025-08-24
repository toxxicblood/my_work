import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import gym
from gym import spaces
import numpy as np
import pandas as pd
from torch.multiprocessing import Process, Pipe
import argparse
import io

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = self.observation_space.sample()

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        reward = np.random.rand()
        self.state = self.observation_space.sample()
        done = np.random.rand() > 0.95
        return self.state, reward, done, {}

def worker(rank, global_model, optimizer):
    local_model = SimpleModel(10, 3).to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = SimpleEnv()
    state = env.reset()
    done = False
    while not done:
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        logits = local_model(state_tensor)
        prob = torch.softmax(logits, dim=-1)
        
        print(f"Rank {rank}, Probabilities: {prob}")

        if torch.isnan(prob).any() or torch.isinf(prob).any() or (prob < 0).any():
            print(f"Rank {rank}, Invalid probabilities found: {prob}")
            return

        action = prob.multinomial(num_samples=1).item()
        
        next_state, reward, done, _ = env.step(action)
        
        # Dummy loss and backprop
        loss = (logits.mean() - reward)**2
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad
        optimizer.step()
        
        local_model.load_state_dict(global_model.state_dict())
        state = next_state

if __name__ == "__main__":
    mp.set_start_method('spawn')
    global_model = SimpleModel(10, 3).to(device)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
    processes = []
    for rank in range(2):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
