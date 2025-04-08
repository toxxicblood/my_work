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
Due to the ability of AI models to find complex relations in data, and the rapid advancement in this field giving the ai even more power, i want to develop a system that understands the market and can work as a professional analyst and portfolio manager.

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

### A Deep reinforcement learning approach for trading optimization in the  forex market with multi-agent Asynchronous Distribution by Davoud Sarani and DR. Parviz Rashidi Khazaee

I intend to compare this paper with the one that follows.FInd it below.
Reference: the full paper is available on my github. [DRL](https://github.com/toxxicblood/learning/blob/main/AI/playground/DRL%20multi%20agent%20for%20different%20timeframes.pdf)

This research is a look into the application of a _multi-agent reinforcement learning framework wit hthe Asynchronous Advantage Actor-Critic algorithm_(__MA+RL+A3C__)

This method employs parallel learning across multiple asynchronous workers, each specialised in trading multiple currency pairs exploring potential or nuanced strategies.

This model outperforms _Proximal Policy Optimisation_

- The problem with having an algorithm trade is the long term delayed rewards.

- In forex data analysis we can either use single or multiple agents to learn the data.
- Here i will focus on using _multi agent_ methods because there is proof that they outperform single agent algorithms:
  - In multi agent algos, the independent agents are specialised in various time periods and they communicate through a hierachical mechanism, aggregating the intelligence across different timeframes to resist noise in financial data and improve trading decisions.
- I can also employ a __supervisory agent__ that selects the most promising trading reccomendations from the independent agents.

There are different parallel multi-agent algorithms like : __A3C, IMPALA and SeedRL.

- In this study, they went with the A3C algorithm because of it's enhanced computational efficiency , reduced training time adn effectiveness in environment exploration learning an iproved policy in less time.
- The A3C algo outperformed the others in training time and profitability.

When implementing only distill profitable trading decisions to students. using distributd computing.

When implementing these models it's good to note that a major drawback or Deep Learning is its time consuming learning mode which could be mitigated by employing distributed cloud computing.

- Another approach is multi-agent DRL algo with trend consisitency regularization.
- Also you can employ teacher agents to diverse sub environments to diversify learned policies , the student agents then utilize profitable knowledge from these teachers to emulate existing trading strategies.
- Reward shaping based on prices for forex trading with DRL using PPO algorithm enhancing agents performance in : profit, sharpe ratio and max drawdown.(market wide approach)

Utilizing DQN and A3C algos with SDAEs and LSTM networks:
    - Specifically the _SDAEs-LSTM A3C model learns a better strategy and surpasses lstm.

- In this research they focused on A3C and PPO to train a RL agent capable of executing trades in the markets

#### Algorithms

__Actor-Critic__:
    - The _actor_ learns a policy(strategy)
    - The _critic_ estimates the expected furture reward and reduces variance by providing a baseline for advantage estimates

__Advantage Function__:
    - this function is used to compute the policy gradient.
    - It quantifies how much better or worse a particular action is compared to the average action.
    - Guides the actor to select actions with better outcomes
    - Adresses credit assignment problem by providing feedback on quality of chosen actions.
    - Calculated as :
        $A(sa) = Q(sa) - V(s)$
        - Where:
          - _Q-Value(action value)_ = expected cumulative reward from a particular action and policy
          - _V-Value(critic value)_ = expected cumulative reward from a particular state onwards using a policy x

__Proximal Policy Optimization(PPO)__:
    - This is an _on-policy RL algorithm_ that optimizes the agent's policy to maximize expected cumulative reward.
    - It iteratively collect data through environment interactions and updating the policy for performance improvement.
    - Basically it optimizes and improves the policy by looking at the state.
    - Maintains a probability distribution over actions for each state repped by a Neural Network.
    - The algo computes the surrogate objective to guide policy gradien for actions with higher returns and the clipped objective to limit policy updates to maintain stable training
    - It adresses theissues of high variance and unstable learning through policy update constraints.
    - It also creates a balance between exploration and exploitation by policy optimization through data.
    - The loss is calculated as follows:
        $𝐿(𝜃)=𝔼𝑡 [𝑚𝑖𝑛(𝑟𝑡(𝜃)𝐴̂𝑡,clip(𝑟𝑡(𝜃),1−𝜖,1+𝜖)𝐴̂𝑡)−𝛽ℋ(𝜋𝜃(⋅|𝑠𝑡))]$
        $𝑟𝑡(𝜃)= 𝜋𝜃(𝑎𝑡|𝑠𝑡)/𝜋𝜃old(𝑎𝑡|𝑠𝑡)$

        Where:
            - __$clip(x,a,b)$__ = a clipping function that clips the value $x$ to the range $[a,b]$.
            - __$ℋ(𝜋𝜃(⋅|𝑠𝑡))$__ = represents entropy of the policy
            - __$𝑟𝑡(𝜃)$__ = ratio of probability of new policy to the old policy.

__Asynchronous Advantage Actor-Critic(A3C)__:
    - This is an advanced variant of the Actor-Critic architecture.
    - The advantage in this algo refres to the advantage function
    - In A3C, multiple local workers run in parallel with their copy of the policy network and environment, collecting experiences and updating global networks asyncrhonously enabling eficient resource utilization, faster convergence , better exploration and more sample-efficient learning.
    - Accumulate gradients with respect to local policy network parameters $𝜃′$ using policy gradient and advantage estimation are calculated as:
        $𝛻𝜃′𝑙𝑜𝑔𝜋(𝑎𝑖|𝑠𝑖;𝜃′)(𝑅−𝑉(𝑠𝑖;𝜃𝑣′))$
    With:
      - $𝜃′v = 𝜕(𝑅−𝑉(𝑆𝑖;𝜃𝑣′))2/𝜕𝜃𝑣$
        - This is how accumulated gradient are calculated with respect to local value network.
        - THis is done using the squared temporal difference error.
        - $(𝑅𝑡 −𝑉(𝑠_𝑡))$ = the difference between the estimated value $V(s_t)$ and the observed reward $R_t$
        - THe $$𝜕/𝜕𝜃_𝑣′$  = the partial derivative with respect to the parameter $𝜃_𝑣′$ and is used to show how small a change in this parameter affects the expression.
      - R is the total discounted return
      - $𝑅 ←𝑟𝑖 +𝛾𝑅$
        - $r_i$ = immediate reward at time step $i$
        - $𝛾$ = discount factor to dicount values of furture rewards
    - The __Policy loss(actor loss)__ for the N time steps in A3C is computed as:
      $𝐿policy =−1/𝑁 *∑^N𝑡=1𝑙𝑜𝑔𝜋(𝑎𝑡|𝑠𝑡)⋅𝐴𝑡$
    - The __Value loss(critic loss)__ guides the value function towards better approximations of the expected return (R).
      - It is the mean squared error between the estimated value function (V) and actual return.
      - THe value loss for N time steps is calculated as:
            $𝐿critic = 1/𝑁∑^N𝑖=1(𝑅𝑖 −𝑉(𝑠𝑖;𝜃𝑣))^2$

    