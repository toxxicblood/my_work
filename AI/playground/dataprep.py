import numpy as np
import pandas as pd
import datetime

def open_file(file):
    df = pd.read_csv(file, parse_dates=['Local time'], index_col="Local time" ,dayfirst=True)

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
def create_windows(features, window_size=16):
    data = []
    for i in range(len(features) - window_size + 1):
        window = features.iloc[i:i+window_size].values
        data.append(window)
    return np.array(data)

# Example usage
# features_df = compute_features(your_candlestick_dataframe)
# windows = create_windows(features_df)
