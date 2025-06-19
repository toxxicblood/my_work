import torch
from A3C import ActorCritic
from gymnasium import ForexEnv
from dataprep import open_file, compute_features
import numpy as np

def evaluate(model, df, window_size=16):
    env = ForexEnv(df, window_size=window_size)
    state = env.reset()
    done = False
    rewards = []
    actions = []
    closes = []
    while not done:
        state_tensor = torch.tensor(state).unsqueeze(0)
        logits, value, *_ = model(state_tensor)
        prob = torch.softmax(logits, dim=-1)
        action = prob.argmax(dim=-1).item()  # Greedy action for evaluation
        next_state, reward, done, _ = env.step1(action)
        rewards.append(reward)
        actions.append(action)
        closes.append(env.df.Close.iloc[env.idx-1])
        state = next_state
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
    }

if __name__ == "__main__":
    # Load model
    model = ActorCritic(input_dim=5, window_size=16, lstm_hidden=128)
    model.load_state_dict(torch.load("a3c_model.pth", map_location="cpu"))  # Save your model after training!
    model.eval()
    # Load and preprocess test data
    df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
    features_df = compute_features(df)
    test_df = features_df.iloc[int(0.8*len(features_df)):]  # Use last 20% for test
    rewards, actions, closes = evaluate(model, test_df)
    metrics = compute_metrics(rewards, closes)
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")