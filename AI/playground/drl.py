import dataprep as dp
import gymnasium
import numpy as np
import pandas as pd

def main():
    # A - Ask
    # B - Bid
    # timeframes = daily, hourly, minute
    #df_DA = dp.open_file('histdata/XAUUSD_Candlestick_1_D_ASK_01.01.2023-22.03.2025.csv')
    #df_DB = dp.open_file('histdata/XAUUSD_Candlestick_1_D_BID_01.01.2023-22.03.2025.csv')
    df_HA = dp.open_file('histdata/XAUUSD_Candlestick_1_Hour_ASK_01.01.2020-22.03.2025.csv')
    #df_HB = dp.open_file('histdata/XAUUSD_Candlestick_1_Hour_BID_01.01.2020-22.03.2025.csv')
    #df_MA = dp.open_file('histdata/XAUUSD_Candlestick_1_M_ASK_01.01.2023-22.03.2025.csv')
    #df_MB = dp.open_file('histdata/XAUUSD_Candlestick_1_M_BID_01.01.2023-22.03.2025.csv')

    #for now i will use the hourly ask data to train the model
    df = df_HA
    print (df)
    train_data = df.loc["2020-01-01":"2024-06-30"]
    test_data = df.loc["2024-07-01":"2025-03-22"]
    train_features = dp.compute_features(train_data)
    test_features = dp.compute_features(test_data)
    print (train_features)
    print (test_features)

    train_windows = dp.create_windows(train_features)
    test_windows = dp.create_windows(test_features)


    # windows = np.array(windows)
    # windows are an NP array.



if __name__ == "__main__":
    main()