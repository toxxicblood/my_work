# FOREX MODEL BY RAMSEY NJIRE

This is a full documentation of me developing my fx model.
Its not gonna be easy but i must make it work.

## What am i trying to solve?

I got into ai because i couldnt control my emotions on the market.
The analysis i did was good, but my emotions got in the way.
My plan is to build an AI model that can:
    - gauge trade sentiment from online articles.
    - Gauge market directions by learning from market data.
For now those are my only goals.

## How do i go about this

1. Find historical forex data that i can train my model on.
2. Develop a model that can use this data to predict market trends.

### Finding the historical data

Im looking for free/ cheap forex historical data from the last 10 - 15 years.
I want to mainly focus on Gold (XAUUSD) and (EURUSD).
I asked chat gpt for sources.
[Histdata](HistData.com)
[Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/)

I got my data from dukascopy

### Data

I downloaded 3 timeframes of data.
Daily, Hourly and minute
THe data is in csv format with 6 categories:
    Local time
    open
    high
    low
    close
    vlume

- Each candle has a specific volume that was traded.

## Ideas

- I can train different models for each timeframe, take their predictions as input into a different model which can take each timeframe and the news sentiment analysis and give a final bias prediction.

## Ideas from documentation

The below are metrics i can use to measure the accuracy of a model i train.
[Accuracy, F1 score, MUC](https://chatgpt.com/c/67ea6594-69b8-800e-b222-85d09937f3b9#:~:text=Accuracy%2C%20F1%20score%2C%20and%20AUC%20are,false%20positives%20and%20false%20negatives%20differ.)
