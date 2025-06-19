from backtesting import Backtest
import pandas as pd
from rl_strategy import RLStrategy

# Load your original OHLCV data (not features)
df = pd.read_csv('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
df['Local time'] = pd.to_datetime(df['Local time'])
df = df.rename(columns={'Local time': 'Date'})
df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
df = df.set_index('Date')

bt = Backtest(df, RLStrategy, cash=10000, commission=0.0)
stats = bt.run()
bt.plot()