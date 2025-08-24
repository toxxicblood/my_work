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
from sklearn.preprocessing import MinMaxScaler

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LSTM_MLP(nn.Module):
    def __init__(self, input_dim, lstm_hidden, fc1_out, fc2_in, fc2_out, fc3_out, output_dim, action_history_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden + action_history_dim, fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)
        self.output_layer = nn.Linear(fc3_out, output_dim)

    def init_hidden(self, batch_size, device):
        return [torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device)]

    def forward(self, x, action_history_tensor, hx=None):
        out, hx = self.lstm(x, hx)
        lstm_out = out[:, -1, :]
        concat_input = torch.cat([lstm_out, action_history_tensor], dim=1)
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.output_layer(h)
        return out, hx

class ActorCritic(nn.Module):
    def __init__(self, input_dim, window_size, lstm_hidden, action_history_dim):
        super().__init__()
        self.actor = LSTM_MLP(input_dim, lstm_hidden, 32, 32, 64, 64, 3, action_history_dim)
        self.critic = LSTM_MLP(input_dim, lstm_hidden, 32, 32, 64, 64, 1, action_history_dim)

    def forward(self, x, action_history_tensor, actor_hx=None, critic_hx=None):
        actor_out, actor_hx = self.actor(x, action_history_tensor, actor_hx)
        critic_out, critic_hx = self.critic(x, action_history_tensor, critic_hx)
        return actor_out, critic_out, actor_hx, critic_hx

class ForexEnv(gym.Env):
    def __init__(self, orig_df, features_df, window_size=16, close_col='Close'):
        super().__init__()
        self.orig_df = orig_df.reset_index(drop=True)
        self.df = features_df.reset_index(drop=True)
        self.window = window_size
        self.close_col = close_col
        self.reset()
        n_features = self.df.shape[1]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (window_size, n_features), dtype=np.float32)

    def reset(self):
        self.idx = self.window
        self.position = 0
        self.total_profit = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        return self.df.iloc[self.idx-self.window:self.idx].values.astype(np.float32)

    def step(self, action):
        if self.idx >= len(self.df):
            return self._get_observation(), 0, True, {}
        price_diff = (self.orig_df[self.close_col].iloc[self.idx] - self.orig_df[self.close_col].iloc[self.idx-1]) / self.orig_df[self.close_col].iloc[self.idx-1]
        transaction_cost = 0.0005
        reward = {0: -price_diff, 1: price_diff, 2: 0}[action] - transaction_cost * (action != self.position)
        self.position = action
        self.idx += 1
        done = self.idx >= len(self.df)
        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.idx}, Position: {self.position}, Total Profit: {self.total_profit}")

    def close(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

def open_file(file_path):
    df = pd.read_csv(file_path)
    return df

def compute_features(df, suffix='', close_col='Close'):
    df = df.copy()
    high_col = f'High{suffix}'
    low_col = f'Low{suffix}'

    df[f'x1{suffix}'] = (df[close_col] - df[close_col].shift(1)) / df[close_col].shift(1)
    df[f'x2{suffix}'] = (df[high_col] - df[high_col].shift(1)) / df[high_col].shift(1)
    df[f'x3{suffix}'] = (df[low_col] - df[low_col].shift(1)) / df[low_col].shift(1)
    df[f'x4{suffix}'] = (df[high_col] - df[close_col]) / df[close_col]
    df[f'x5{suffix}'] = (df[close_col] - df[low_col]) / df[close_col]
    return df[[f'x1{suffix}', f'x2{suffix}', f'x3{suffix}', f'x4{suffix}', f'x5{suffix}']]

def worker(rank, global_model, optimizer, orig_df, features_df, window_size, gamma, lstm_hidden, close_col='Close'):
    local_model = ActorCritic(input_dim=features_df.shape[1], window_size=window_size, lstm_hidden=lstm_hidden, action_history_dim=window_size * 3).to(device)
    local_model.load_state_dict(global_model.state_dict())
    env = ForexEnv(orig_df, features_df, window_size=window_size, close_col=close_col)
    state = env.reset()
    done = False
    action_history = [2] * window_size
    while not done:
        log_probs, values, rewards = [], [], []
        hx_actor, hx_critic = None, None
        for _ in range(20):
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            action_hist_encoded = np.zeros((len(action_history), 3))
            for i, act in enumerate(action_history):
                action_hist_encoded[i, act] = 1
            action_hist_tensor = torch.from_numpy(action_hist_encoded.flatten()).unsqueeze(0).float().to(device)
            logits, value, hx_actor, hx_critic = local_model(state_tensor, action_hist_tensor, hx_actor, hx_critic)
            prob = torch.softmax(logits, dim=-1)
            action = prob.multinomial(num_samples=1).item()
            action_history.append(action)
            if len(action_history) > window_size:
                action_history.pop(0)
            log_prob = torch.log(prob.squeeze(0)[action] + 1e-10)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(reward)
            state = next_state
            if done:
                break
        R = local_model(torch.from_numpy(state).unsqueeze(0).to(device), action_hist_tensor)[1].item()
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        values = torch.stack(values)
        advantage = returns - values
        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=-1).mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 0.5)
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A3C Hyperparameter Tuning')
    parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of training workers')
    parser.add_argument('--model_name', type=str, default='a3c_model.pth', help='name for the saved model')
    parser.add_argument('--data_config', type=str, default='ask', choices=['ask', 'ask_bid', 'multi_timeframe'], help='data configuration')
    args = parser.parse_args()

    mp.set_start_method('spawn')

    close_col = 'Close'
    if args.data_config == 'ask':
        df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
        df['Local time'] = pd.to_datetime(df['Local time'], format='mixed', utc=True)
        df = df.set_index('Local time')
        df.sort_index(inplace=True)
        features_df = compute_features(df)
        input_dim = 5
    elif args.data_config == 'ask_bid':
        ask_df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
        bid_df = open_file('histdata/XAUUSD_Candlestick_1_Hour_BID_01.01.2020-22.03.2025.csv')
        ask_df['Local time'] = pd.to_datetime(ask_df['Local time'], format='mixed', utc=True)
        bid_df['Local time'] = pd.to_datetime(bid_df['Local time'], format='mixed', utc=True)
        ask_df = ask_df.set_index('Local time')
        bid_df = bid_df.set_index('Local time')
        ask_df.sort_index(inplace=True)
        bid_df.sort_index(inplace=True)
        
        ask_df.rename(columns=lambda x: x + '_ask', inplace=True)
        bid_df.rename(columns=lambda x: x + '_bid', inplace=True)

        ask_features = compute_features(ask_df, suffix='_ask', close_col='Close_ask')
        bid_features = compute_features(bid_df, suffix='_bid', close_col='Close_bid')
        
        features_df = pd.concat([ask_features, bid_features], axis=1)
        features_df.dropna(inplace=True)
        df = ask_df # Use ask for original prices
        input_dim = 10
        close_col = 'Close_ask'
    elif args.data_config == 'multi_timeframe':
        daily_df = open_file('histdata/XAUUSD_Candlestick_1_D_ASK_01.01.2023-22.03.2025.csv')
        hourly_df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
        minute_df = open_file('histdata/XAUUSD_Candlestick_1_M_ASK_01.01.2023-22.03.2025.csv')

        for df_item in [daily_df, hourly_df, minute_df]:
            df_item['Local time'] = pd.to_datetime(df_item['Local time'], format='mixed', utc=True)
            df_item = df_item.set_index('Local time')
            df_item.sort_index(inplace=True)

        hourly_df.rename(columns=lambda x: x + '_hourly', inplace=True)
        daily_df.rename(columns=lambda x: x + '_daily', inplace=True)
        minute_df.rename(columns=lambda x: x + '_minute', inplace=True)

        merged_df = pd.merge(hourly_df, daily_df, left_index=True, right_index=True, how='left')
        merged_df = pd.merge(merged_df, minute_df, left_index=True, right_index=True, how='left')
        merged_df.fillna(method='ffill', inplace=True)
        merged_df.dropna(inplace=True)

        daily_features = compute_features(merged_df, suffix='_daily', close_col='Close_daily')
        hourly_features = compute_features(merged_df, suffix='_hourly', close_col='Close_hourly')
        minute_features = compute_features(merged_df, suffix='_minute', close_col='Close_minute')

        features_df = pd.concat([daily_features, hourly_features, minute_features], axis=1)
        features_df.dropna(inplace=True)
        df = hourly_df # Use hourly for original prices
        input_dim = 15
        close_col = 'Close_hourly'

    features_df.dropna(inplace=True)
    scaler = MinMaxScaler()
    features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)

    common_index = df.index.intersection(features_df.index)
    df = df.loc[common_index]
    features_df = features_df.loc[common_index]

    train_orig = df.loc['2020-01-01':'2024-06-30']
    test_orig = df.loc['2024-07-01':'2025-03-21']
    train_feat = features_df.loc[train_orig.index]
    test_feat = features_df.loc[test_orig.index]

    global_model = ActorCritic(input_dim=input_dim, window_size=16, lstm_hidden=args.hidden_size, action_history_dim=16*3).to(device)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr)
    processes = []
    print(f"Starting training for model: {args.model_name}")
    for rank in range(args.num_workers):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer, train_orig, train_feat, 16, args.gamma, args.hidden_size, close_col))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(global_model.state_dict(), args.model_name)
    print(f"Training complete. Model saved as {args.model_name}")