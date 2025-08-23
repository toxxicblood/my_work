import torch
from train import ActorCritic, ForexEnv, open_file, compute_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate(model, orig_df, features_df, window_size=16):
    env = ForexEnv(orig_df, features_df, window_size=window_size)
    state = env.reset()
    done = False
    rewards = []
    actions = []
    closes = []
    action_history = [2] * window_size
    while not done:
        state_tensor = torch.tensor(state).unsqueeze(0)
        action_hist_encoded = np.zeros((len(action_history), 3))
        for i, act in enumerate(action_history):
            action_hist_encoded[i, act] = 1
        action_hist_tensor = torch.tensor(action_hist_encoded.flatten()).unsqueeze(0).float()
        logits, value, *_ = model(state_tensor, action_hist_tensor)
        prob = torch.softmax(logits, dim=-1)
        action = prob.argmax(dim=-1).item()  # Greedy action for evaluation
        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        idx = env.idx - 1
        # Only append if idx is within bounds
        if 0 <= idx < len(env.orig_df):
            closes.append(env.orig_df['Close'].iloc[idx])
        state = next_state
        action_history.append(action)
        if len(action_history) > window_size:
            action_history.pop(0)
    return rewards, actions, closes

def compute_metrics(rewards, closes):
    returns = np.array(rewards)
    total_return = (closes[-1] - closes[0]) / closes[0] * 100 if closes else 0
    sharpe = returns.mean() / (returns.std() + 1e-8)
    profit_factor = returns[returns > 0].sum() / (-returns[returns < 0].sum() + 1e-8)
    # Max Drawdown
    equity_curve = np.cumsum(returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve)
    max_drawdown = drawdown.max()
    return {
        "Total Return (%)": total_return,
        "Sharpe Ratio": sharpe,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_drawdown
    }, equity_curve

if __name__ == "__main__":
    model_files = [f for f in os.listdir('.') if f.startswith('a3c_model') and f.endswith('.pth')]
    
    df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
    df['Local time'] = pd.to_datetime(df['Local time'], format='mixed')
    df = df.set_index('Local time')
    df.sort_index(inplace=True)
    features_df = compute_features(df)
    test_orig = df.loc['2024-07-01':'2025-03-21'].reset_index(drop=True)
    test_feat = features_df.loc['2024-07-01':'2025-03-21'].reset_index(drop=True)

    for model_file in model_files:
        print(f"--- Evaluating {model_file} ---")
        # Extract hidden_size from model_name, default to 128
        hidden_size = 128
        if '_hs_' in model_file:
            try:
                hidden_size = int(model_file.split('_hs_')[1].split('.pth')[0])
            except (IndexError, ValueError):
                pass

        model = ActorCritic(input_dim=5, window_size=16, lstm_hidden=hidden_size, action_history_dim=16*3)
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.eval()
        
        rewards, actions, closes = evaluate(model, test_orig, test_feat)
        metrics, equity_curve = compute_metrics(rewards, closes)
        
        print("Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Plot and save the equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve)
        plt.title(f"Equity Curve - {model_file}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Reward")
        plt.savefig(f"equity_curve_{model_file.replace('.pth', '.png')}")
        print(f"Equity curve plot saved to equity_curve_{model_file.replace('.pth', '.png')}")
        print("-" * 30)