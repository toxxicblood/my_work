
import comet_ml
import torch
import torch.nn.functional as F
import torch.nn as nn
import mitdeeplearning as mdl
#from google.colab import files
import numpy as np
import os
import torch.optim as optim
import time
from IPython import display as ipythondisplay
from tqdm import tqdm
"""from google.colab import files"""
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import gym
from gym import spaces

COMET_API_KEY =  "fux8OQ14wngoM7Cmu6HRqiGQa" # Replace with your key or set as env var
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###hyperparameter setting and optimisation
params = dict(
    window_size = 16, #experiment between 1 and 64
    gamma = 0.99,
    learning_rate = 4e-5,
    training_iterations = 100 ,# increase for longer training
    rollout_steps = 20,
    hidden_size = 1024,#experiment between 1 and 2048
    input_dim = 5,

)


#data preparation functions:
def open_file(file):
    df = pd.read_csv(file, )#parse_dates=['Local time'], index_col="Local time" ,dayfirst=True)
    datetime_values = []
    # Convert 'Local time' to datetime objects
    for value in df['Local time']:
        converted = None
        for fmt in ['%d.%m.%Y %H:%M:%S.%f', '%d.%m.%Y %H:%M:%S', '%Y.%m.%d %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S']:
            try:
                converted = pd.to_datetime(value, format=fmt)
                break
            except ValueError:
                pass
        datetime_values.append(converted)
    #df['Local time'] = pd.to_datetime(df['Local time'], dayfirst = True)
    #df = df.sort_values(by='Local time').dropna()
    return df

# Assume df is a DataFrame with columns: 'open', 'High', 'Low', 'Close'
def compute_features(df):
    df = df.copy()
    df.loc[:,'x1'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df.loc[:,'x2'] = (df['High'] - df['High'].shift(1)) / df['High'].shift(1)
    df.loc[:,'x3'] = (df['Low'] - df['Low'].shift(1)) / df['Low'].shift(1)
    df.loc[:,'x4'] = (df['High'] - df['Close']) / df['Close']
    df.loc[:,'x5'] = (df['Close'] - df['Low']) / df['Close']
    df.dropna(inplace=True)
    return df[['x1', 'x2', 'x3', 'x4', 'x5']]

# Build sliding window data: returns a 3D array [batch_size, window_size, features]
def create_windows(features, window_size):
    data = []
    for i in range(len(features) - window_size + 1):
        window = features.iloc[i:i+window_size].values
        data.append(window)
    return np.array(data)
# Example usage
# features_df = compute_features(your_candlestick_dataframe)
# windows = create_windows(features_df)

#loading data
# A - Ask
# B - Bid
# timeframes = daily, hourly, minute
#df_DA = dp.open_file('histdata/XAUUSD_Candlestick_1_D_ASK_01.01.2023-22.03.2025.csv')
#df_DB = dp.open_file('histdata/XAUUSD_Candlestick_1_D_BID_01.01.2023-22.03.2025.csv')
df_HA = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
#df_HB = dp.open_file('histdata/XAUUSD_Candlestick_1_Hour_BID_01.01.2020-22.03.2025.csv')
#df_MA = dp.open_file('histdata/XAUUSD_Candlestick_1_M_ASK_01.01.2023-22.03.2025.csv')
#df_MB = dp.open_file('histdata/XAUUSD_Candlestick_1_M_BID_01.01.2023-22.03.2025.csv')

#for now i will use the hourly ask data to train the model
#df = files.upload()
df = df_HA
print(df[:100])
train_data = df.loc["2020-01-01":"2024-06-30"]
test_data = df.loc["2024-07-01":"2025-03-22"]
train_features = compute_features(train_data)
test_features = compute_features(test_data)


train_windows = create_windows(train_features, params['window_size'])
test_windows = create_windows(test_features, params['window_size'])

print("Train windows shape:", train_windows.shape)
print("train features shape:", train_features.shape)
#print("feature shape:", train_features.shape)
# windows = np.array(windows)
# windows are an NP array.
###batch defined to create training examples
"""def get_batch(vectorized_songs, seq_length, batch_size):
    #len o vectorized songs str
    n = vectorized_songs.shape[0] -1
    idx = np.random.choice(n - seq_length, batch_size)

    #input seq list for training bathc
    input_batch = np.array([vectorized_songs[i:i + seq_length] for i in idx])
    #target seq list for training batch
    output_batch = np.array([vectorized_songs[i + 1:i + seq_length + 1] for i in idx])

    #convert hese batches to tensors
    x_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
    y_batch = torch.tensor(output_batch, dtype=torch.long, device=device)

    return x_batch, y_batch"""

class ForexEnv (gym.Env):
    def __init__(self, orig_df, features_df, window_size=16):
        super().__init__()
        self.orig_df = orig_df.reset_index(drop=True)
        self.df = features_df.reset_index(drop=True)
        self.window = window_size
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

    def step1(self, action):
        if self.idx >= len(self.df):
            return self._get_observation(), 0, True, {}
        price_diff = (self.orig_df['Close'].iloc[self.idx] - self.orig_df['Close'].iloc[self.idx-1]) / self.orig_df['Close'].iloc[self.idx-1]
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

    #normalise the reward function as described
    #make sure env supports long, short and neutral

### defining the neural network:
class LSTM_MLP(nn.Module):
    def __init__(self, input_dim, lstm_hidden, fc1_out, fc2_in, fc2_out, fc3_out, output_dim, action_history_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden + action_history_dim, fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)
        self.output_layer = nn.Linear(fc3_out, output_dim)

    def init_hidden(self, batch_size, device):
        #initialize hidden state and cell sstate to zeros
        return [torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device)]


    """    def forward(self, x, hx=None):
            out, hx = self.lstm(x, hx)
            h = F.relu(self.fc1(out[:, -1, :]))
            # Concatenate with zeros to match fc2's input size if needed
            if h.shape[1] < self.fc2.in_features:
                pad = torch.zeros(h.shape[0], self.fc2.in_features - h.shape[1], device=h.device)
                h = torch.cat([h, pad], dim=1)
            h = F.relu(self.fc2(h))
            h = F.relu(self.fc3(h))
            out = self.output_layer(h)
            return out, hx
    """
    def forward(self, x, action_history_tensor, hx=None):
        out, hx = self.lstm(x, hx)
        lstm_out = out[:, -1, :]  # take last timestep
        concat_input = torch.cat([lstm_out, action_history_tensor], dim=1)  # concat with action history
        h = F.relu(self.fc1(concat_input))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.output_layer(h)
        return out, hx
"""class ActorCritic(nn.Module):
    def __init__(self, input_dim, window_size, lstm_hidden):
        super().__init__()
        # Actor: output_dim=3, Critic: output_dim=1
        self.actor = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 3)
        self.critic = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 1)

    def forward(self, x, actor_hx=None, critic_hx=None):
        actor_out, actor_hx = self.actor(x, actor_hx)
        critic_out, critic_hx = self.critic(x, critic_hx)
        return actor_out, critic_out, actor_hx, critic_hx"""
class ActorCritic(nn.Module):
    def __init__(self, input_dim, window_size, lstm_hidden, action_history_dim):
        super().__init__()
        self.actor = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 3, action_history_dim)
        self.critic = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 1, action_history_dim)

    def forward(self, x, action_history_tensor, actor_hx=None, critic_hx=None):
        actor_out, actor_hx = self.actor(x, action_history_tensor, actor_hx)
        critic_out, critic_hx = self.critic(x, action_history_tensor, critic_hx)
        return actor_out, critic_out, actor_hx, critic_hx


cross_entropy = nn.CrossEntropyLoss()
def compute_loss(labels, logits):
    """
    Inputs:
      labels: (batch_size, sequence_length)
      logits: (batch_size, sequence_length, vocab_size)

    Output:
      loss: scalar cross entropy loss over the batch and sequence length
    """
    #batch labels so theat their shape is (b*L)
    batched_labels = labels.view(-1)

    #batch logits so that their shape is (B*, V)
    batched_logits = logits.view(-1, logits.size(-1))

    #compute cross entropy loss using batched nex chars and predictions
    loss = cross_entropy(batched_logits, batched_labels)
    return loss


#checkpoint location
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
os.makedirs(checkpoint_dir, exist_ok=True)

def encode_action_history(action_history, num_actions=3):
    # Convert list of action indices into one-hot encoded tensor
    one_hot = np.zeros((len(action_history), num_actions))
    for i, act in enumerate(action_history):
        one_hot[i, act] = 1
    return one_hot.flatten()  # Flatten to single vector

#create experiment to track our training
def create_experiment():
    #end any prior experiments
    if 'experiment' in locals():
        experiment.end()
        #locals()['experiment'].end()

    #initiate the comet experient
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name = "forex-model-py"
    )
    # log our hyperparams
    for param, value in params.items():
        experiment.log_parameter(param, value)
    experiment.flush()

    return experiment
# action_history_dim = window_size * num_actions
action_history_dim = params['window_size'] * 3  # 3 possible actions
model = ActorCritic(
    input_dim= params['input_dim'],
    window_size=params['window_size'],
    lstm_hidden= params['hidden_size'],
    action_history_dim=action_history_dim
)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

def train_step(x,y):
    model.train()
    optimizer.zero_grad()
    y_hat = model(x)
    loss = compute_loss(y, y_hat)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max=5.0)#gradient clipping
    optimizer.step()
    return loss

###model training
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='loss')
experiment = create_experiment()

best_loss = float('inf')
best_ckpt_path = os.path.join(checkpoint_dir, 'best_model.pth')

if hasattr(tqdm, '_instances'): tqdm._instances.clear()
for episode in tqdm(range(params['training_iterations'])):
    """x_batch, y_batch = get_batch(train_data, params['seq_length'], params['batch_size'])

    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    #take a train step"""
    # loss = train_step(x_batch, y_batch)

    env = ForexEnv(train_windows, window_size=params['window_size'])
    state = env.reset()
    done = False
    episode_loss = []
    action_history = [2] * params['window_size']  # Store action history for encoding
    while not done:
        log_probs, valuse, rewards = [],[],[]
        hx_actor, hx_critic = None, None
        for _ in range(params['rollout_steps']):
            state_tensor = torch.tensor(state).unsqueeze(0).float().to(device)
            action_hist_encoded = encode_action_history(action_history)
            action_hist_tensor = torch.tensor(action_hist_encoded).unsqueeze(0).float().to(device)

            logits, value, hx_actor, hx_critic = model(state_tensor,action_hist_tensor, hx_actor, hx_critic)
            prob = torch.softmax(logits, dim=-1)
            action = prob.multinomial(num_samples=1).item()
            action_history.append(action)
            if len(action_history) > params['window_size']:
                action_history.pop(0)
            log_prob = torch.log(prob.squeeze(0)[action])
            next_state, reward, done, _ = env.step1(action)
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(reward)
            state = next_state
            if done:
                break
        # Compute returns and advantage
        R = 0 if done else model(torch.tensor(state).unsqueeze(0))[1].item()
        returns = []
        for r in reversed(rewards):
            R = r + params['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        values = torch.stack(values)
        advantage = returns - values
        # Compute losses
        policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=-1).mean()  # entropy regularization
        loss = policy_loss + 0.5 * value_loss  - 0.01 * entropy  # combine losses with entropy regularization
        #update progress bar and visualise within notebook
        history.append(loss.item())
        plotter.plot(history)
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_loss.append(loss.item())
        if episode % 10 == 0:
            experiment.log_metric("loss", loss.item(), step=iter)

            #save regular model checkpoint
        if iter % 100 ==0:
            torch.save(model.state_dict(), checkpoint_prefix)

        #save best checkpoint if loss improves
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved at iteration {iter} with loss {best_loss:.4f}")
    avg_loss = np.mean(episode_loss)
    history.append(avg_loss)
    experiment.log_metric("loss", avg_loss, step=episode)
    plotter.plot(history)
    print(f"Episode {episode+1}/{params['training_iterations']}, Loss: {avg_loss:.4f}")
# Save model
torch.save(model.state_dict(), "a3c_model_colab.pth")
experiment.flush()
print("Training complete and model saved.")

"""    #log the loss to the comet interface






#save the final trained model
torch.save(model.state_dict(), checkpoint_prefix)
#save model state dict:
"""
