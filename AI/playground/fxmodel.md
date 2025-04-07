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

## Research

I went through a couple of research papers to see how other people have used AI and ML in forex market analysis.
In the following sections i will be breaking down my findings and thus create a hypothesis of the model i will go ahead with testing the various methodologies experimenting with different combinaions till i find one that performs well.

THe models i found being used most were LSTM(long short term memory) algorithms and CNNs(convolutional neural networks).
THe approach most used was deep learning.
In the following sections is a summary of my findings with the respective papers.

### A Deep reinforcement learning approach for trading optimization in the  forex market with multi-agent Asynchronous Distribution by Davoud Sarani and DR. Parviz Rashidi Khazaee.

I intend to compare this paper with the one that follows.FInd it below.
Reference: the full paper is available on my github. [DRL](https://github.com/toxxicblood/learning/blob/main/AI/playground/DRL%20multi%20agent%20for%20different%20timeframes.pdf)