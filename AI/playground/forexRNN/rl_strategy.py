from backtesting import Strategy
import torch
from train import ActorCritic, compute_features
import numpy as np

class RLStrategy(Strategy):
    def init(self):
        # Load your trained model
        self.model = ActorCritic(input_dim=5, window_size=16, lstm_hidden=128, action_history_dim=16*3)
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
        action_history = [2] * self.window_size
        action_hist_encoded = np.zeros((len(action_history), 3))
        for i, act in enumerate(action_history):
            action_hist_encoded[i, act] = 1
        action_hist_tensor = torch.tensor(action_hist_encoded.flatten()).unsqueeze(0).float()
        logits, _, *_ = self.model(state_tensor, action_hist_tensor)
        action = torch.softmax(logits, dim=-1).argmax(dim=-1).item()
        # 0: short, 1: long, 2: hold
        if action == 1 and not self.position.is_long:
            self.buy()
        elif action == 0 and not self.position.is_short:
            self.sell()
        elif action == 2:
            self.position.close()