from backtesting import Strategy
import torch
from A3C import ActorCritic
from dataprep import compute_features

class RLStrategy(Strategy):
    def init(self):
        # Load your trained model
        self.model = ActorCritic(input_dim=5, window_size=16, lstm_hidden=128)
        self.model.load_state_dict(torch.load("a3c_model.pth", map_location="cpu"))
        self.model.eval()
        # Prepare features
        features = compute_features(self.data.df)
        self.features = features.values.astype('float32')
        self.window_size = 16

    def next(self):
        idx = len(self.data.Close) - 1
        if idx < self.window_size:
            return
        state = self.features[idx - self.window_size + 1:idx + 1]
        state_tensor = torch.tensor(state).unsqueeze(0)
        logits, _, *_ = self.model(state_tensor)
        action = torch.softmax(logits, dim=-1).argmax(dim=-1).item()
        # 0: short, 1: long, 2: hold
        if action == 1 and not self.position.is_long:
            self.buy()
        elif action == 0 and not self.position.is_short:
            self.sell()
        elif action == 2:
            self.position.close()