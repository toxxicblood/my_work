import torch
from train import ActorCritic, ForexEnv, open_file, compute_features
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

def evaluate(model, orig_df, features_df, window_size=16, close_col='Close'):
    env = ForexEnv(orig_df, features_df, window_size=window_size, close_col=close_col)
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
            closes.append(env.orig_df[close_col].iloc[idx])
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
    parser = argparse.ArgumentParser(description='A3C Evaluation')
    parser.add_argument('--model_name', type=str, required=True, help='name of the saved model to evaluate')
    parser.add_argument('--data_config', type=str, default='ask', choices=['ask', 'ask_bid', 'multi_timeframe'], help='data configuration')
    args = parser.parse_args()

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
            df_item.set_index('Local time', inplace=True)
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

    common_index = df.index.intersection(features_df.index)
    df = df.loc[common_index]
    features_df = features_df.loc[common_index]
    
    test_orig = df.loc['2024-07-01':'2025-03-21'].reset_index(drop=True)
    test_feat = features_df.loc['2024-07-01':'2025-03-21'].reset_index(drop=True)

    hidden_size = 128
    if '_hs_' in args.model_name:
        try:
            hidden_size = int(args.model_name.split('_hs_')[1].split('_')[0])
        except (IndexError, ValueError):
            pass

    model = ActorCritic(input_dim=input_dim, window_size=16, lstm_hidden=hidden_size, action_history_dim=16*3)
    model.load_state_dict(torch.load(args.model_name, map_location="cpu"))
    model.eval()
    
    rewards, actions, closes = evaluate(model, test_orig, test_feat, close_col=close_col)
    metrics, equity_curve = compute_metrics(rewards, closes)
    
    print(f"--- Evaluation for {args.model_name} ---")
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot and save the equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title(f"Equity Curve - {args.model_name}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Reward")
    plt.savefig(f"equity_curve_{args.model_name.replace('.pth', '.png')}")
    print(f"Equity curve plot saved to equity_curve_{args.model_name.replace('.pth', '.png')}")
    print("-" * 30)
