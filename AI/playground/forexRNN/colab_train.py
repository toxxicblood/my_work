import comet_ml
import torch
import torch.nn.functional as F
from model import ActorCritic
from env import ForexEnv
from drl import open_file, compute_features, create_windows
import mitdeeplearning as mdl
import numpy as np
import os
import torch.optim as optim
import time
from IPython import display as ipythondisplay
from tqdm import tqdm
from google.colab import files
import pandas as pd
import datetime


# Set your Comet API key
COMET_API_KEY =  "fux8OQ14wngoM7Cmu6HRqiGQa" # Replace with your key or set as env var
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#loading data



def train_colab():
    experiment = comet_ml.Experiment(
        api_key=COMET_API_KEY,
        project_name="forex_rl_colab"
    )

    # Hyperparameters
    window_size = 16
    gamma = 0.99
    lr = 4e-5
    num_episodes = 10  # Increase for longer training
    rollout_steps = 20

    # Data
    df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
    features_df = compute_features(df)
    train_df = features_df.iloc[:int(0.8*len(features_df))]

    # Model and optimizer
    model = ActorCritic(input_dim=5, window_size=window_size, lstm_hidden=128, action_history_dim=16*3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Plotter
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Episodes', ylabel='Loss')
    history = []

    for episode in range(num_episodes):
        env = ForexEnv(train_df, window_size=window_size)
        state = env.reset()
        done = False
        episode_loss = []
        action_history = [2] * window_size
        while not done:
            log_probs, values, rewards = [], [], []
            hx_actor, hx_critic = None, None
            for _ in range(rollout_steps):
                state_tensor = torch.tensor(state).unsqueeze(0)
                action_hist_encoded = np.zeros((len(action_history), 3))
                for i, act in enumerate(action_history):
                    action_hist_encoded[i, act] = 1
                action_hist_tensor = torch.tensor(action_hist_encoded.flatten()).unsqueeze(0).float()
                logits, value, hx_actor, hx_critic = model(state_tensor, action_hist_tensor, hx_actor, hx_critic)
                prob = torch.softmax(logits, dim=-1)
                action = prob.multinomial(num_samples=1).item()
                log_prob = torch.log(prob.squeeze(0)[action])
                next_state, reward, done, _ = env.step(action)
                log_probs.append(log_prob)
                values.append(value.squeeze(0))
                rewards.append(reward)
                state = next_state
                action_history.append(action)
                if len(action_history) > window_size:
                    action_history.pop(0)
                if done:
                    break
            # Compute returns and advantage
            R = 0 if done else model(torch.tensor(state).unsqueeze(0), action_hist_tensor)[1].item()
            returns = []
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            values = torch.stack(values)
            advantage = returns - values
            # Compute losses
            policy_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            loss = policy_loss + 0.5 * value_loss
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            episode_loss.append(loss.item())
        avg_loss = np.mean(episode_loss)
        history.append(avg_loss)
        experiment.log_metric("loss", avg_loss, step=episode)
        plotter.plot(history)
        print(f"Episode {episode+1}/{num_episodes}, Loss: {avg_loss:.4f}")
    # Save model
    torch.save(model.state_dict(), "a3c_model_colab.pth")
    experiment.flush()
    print("Training complete and model saved.")

if __name__ == "__main__":
    train_colab()