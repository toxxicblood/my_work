from backtesting import Backtest, Strategy
import torch

class RLStrategy(Strategy):
    def __init__(self):
        # Initialize indicators or other variables here
        pass

    def next(self):
        # Implement your trading logic here
        # Example: self.buy() or self.sell()
        obs = self.data.df[self.i-16:self.i].values
        logits, _, _ = model(torch.tensor(obs[None], dtype=torch.float32))
        action = torch.argmax(logits, dim=-1).item()
        if action == 1: self.buy()
        elif action == 0: self.sell()

    bt = Backtest(test_df, RLStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    bt.plot()
