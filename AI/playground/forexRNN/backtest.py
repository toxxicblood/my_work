import numpy as np
import pandas as pd
import torch

class Backtest:
    def __init__(self, orig_df, features_df, model, window_size=16, initial_cash=10000, commission=0.0005, position_size=0.1, close_col='Close'):
        self.orig_df = orig_df
        self.features_df = features_df
        self.model = model
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.commission = commission
        self.position_size = position_size
        self.close_col = close_col

    def run(self):
        cash = self.initial_cash
        position = 0
        equity = [self.initial_cash]
        action_history = [2] * self.window_size
        
        for i in range(self.window_size, len(self.features_df)):
            state = self.features_df.iloc[i-self.window_size:i].values
            
            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            action_hist_encoded = np.zeros((len(action_history), 3))
            for j, act in enumerate(action_history):
                action_hist_encoded[j, act] = 1
            action_hist_tensor = torch.from_numpy(action_hist_encoded.flatten()).unsqueeze(0).float()
            
            logits, _, *_ = self.model(state_tensor, action_hist_tensor)
            prob = torch.softmax(logits, dim=-1)
            action = prob.argmax(dim=-1).item()

            action_history.append(action)
            if len(action_history) > self.window_size:
                action_history.pop(0)

            # Execute trade
            if action == 1 and position == 0: # Buy
                position = cash * self.position_size / self.orig_df[self.close_col].iloc[i]
                cash -= position * self.orig_df[self.close_col].iloc[i] * (1 + self.commission)
            elif action == 0 and position > 0: # Sell
                cash += position * self.orig_df[self.close_col].iloc[i] * (1 - self.commission)
                position = 0
            
            # Update equity
            equity.append(cash + position * self.orig_df[self.close_col].iloc[i])

        return pd.Series(equity, index=self.orig_df.index[self.window_size-1:])

def compute_metrics(equity_curve):
    returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0] * 100
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * 24) # Annualized Sharpe for hourly data
    
    # Max Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    
    return {
        "Total Return (%)": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }
