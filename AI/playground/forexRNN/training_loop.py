import torch
import torch.multiprocessing as mp
from drl import ActorCritic
from drl import ForexEnv
from drl import open_file, compute_features
import pandas as pd
from torch.multiprocessing import Process, Pipe


def worker(rank, global_model, optimizer, orig_df, features_df, window_size=16, gamma=0.99):
    local_model = ActorCritic(input_dim=5, window_size=window_size, lstm_hidden=128)
    local_model.load_state_dict(global_model.state_dict())
    env = ForexEnv(orig_df, features_df, window_size=window_size)
    state = env.reset()
    done = False
    while not done:
        # Collect rollout
        log_probs, values, rewards = [], [], []
        hx_actor, hx_critic = None, None
        for _ in range(20):  # n-step
            state_tensor = torch.tensor(state).unsqueeze(0)
            logits, value, hx_actor, hx_critic = local_model(state_tensor, hx_actor, hx_critic)
            prob = torch.softmax(logits, dim=-1)
            action = prob.multinomial(num_samples=1).item()
            log_prob = torch.log(prob.squeeze(0)[action])
            next_state, reward, done, _ = env.step1(action)
            log_probs.append(log_prob)
            values.append(value.squeeze(0))
            rewards.append(reward)
            state = next_state
            if done:
                break
        # Compute returns and advantage
        R = 0 if done else local_model(torch.tensor(state).unsqueeze(0))[1].item()
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
        # Backprop and update global model
        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())



if __name__ == "__main__":
    mp.set_start_method('spawn')
    # --- Load and preprocess data ---
    df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
    features_df = compute_features(df)
    # Ensure the index is datetime for slicing
    if not isinstance(features_df.index, pd.DatetimeIndex):
        if 'Local time' in df.columns:
            # Robustly parse mixed datetime formats
            parsed_dates = pd.to_datetime(df['Local time'].iloc[-len(features_df):], errors='coerce', dayfirst=True)
            features_df.index = parsed_dates
            # Drop rows where datetime parsing failed
            features_df = features_df[features_df.index.notnull()]
        else:
            raise ValueError("No datetime index or 'Local time' column found for slicing.")

    # Train/test split by date (on original df)
    train_orig = df.loc['2020-01-01':'2024-06-30']
    test_orig = df.loc['2024-07-01':'2025-03-21']
    train_feat = compute_features(train_orig)
    test_feat = compute_features(test_orig)

    # --- Initialize model and optimizer ---
    global_model = ActorCritic(input_dim=5, window_size=16, lstm_hidden=128)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=4e-5)
    # --- Start workers ---
    processes = []
    num_workers = 5
    for rank in range(num_workers):
        p = mp.Process(target=worker, args=(rank, global_model, optimizer, train_orig, train_feat))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    torch.save(global_model.state_dict(), "a3c_model.pth")

