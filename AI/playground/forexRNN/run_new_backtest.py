import torch
from train import ActorCritic, open_file, compute_features
from backtest import Backtest, compute_metrics
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import re

if __name__ == "__main__":
    model_files = [f for f in os.listdir('.') if f.startswith('a3c_model') and f.endswith('.pth')]

    for model_file in model_files:
        print(f"--- Backtesting {model_file} ---")
        
        hidden_size = 128
        match = re.search(r'_hs_(\d+)', model_file)
        if match:
            hidden_size = int(match.group(1))

        data_config = 'ask'
        if 'ask_bid' in model_file:
            data_config = 'ask_bid'
        elif 'multi_timeframe' in model_file:
            data_config = 'multi_timeframe'

        close_col = 'Close'
        if data_config == 'ask':
            df = open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
            df['Local time'] = pd.to_datetime(df['Local time'], format='mixed', utc=True)
            df = df.set_index('Local time')
            df.sort_index(inplace=True)
            features_df = compute_features(df)
            input_dim = 5
        elif data_config == 'ask_bid':
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
            df = ask_df
            input_dim = 10
            close_col = 'Close_ask'
        elif data_config == 'multi_timeframe':
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
            df = hourly_df
            input_dim = 15
            close_col = 'Close_hourly'

        features_df.dropna(inplace=True)
        scaler = MinMaxScaler()
        features_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns, index=features_df.index)

        common_index = df.index.intersection(features_df.index)
        df = df.loc[common_index]
        features_df = features_df.loc[common_index]

        test_orig = df.loc['2024-07-01':'2025-03-21']
        test_feat = features_df.loc[test_orig.index]

        model = ActorCritic(input_dim=input_dim, window_size=16, lstm_hidden=hidden_size, action_history_dim=16*3)
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        model.eval()

        backtest = Backtest(test_orig, test_feat, model, close_col=close_col)
        equity_curve = backtest.run()
        
        metrics = compute_metrics(equity_curve)
        
        print("Backtest Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        equity_curve.plot(figsize=(12, 6))
        plt.title(f"Equity Curve - {model_file}")
        plt.xlabel("Time")
        plt.ylabel("Equity")
        plt.savefig(f"new_equity_curve_{model_file.replace('.pth', '.png')}")
        print(f"New equity curve plot saved to new_equity_curve_{model_file.replace('.pth', '.png')}")
        print("-" * 30)