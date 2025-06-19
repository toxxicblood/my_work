import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_MLP(nn.Module):
    def __init__(self, input_dim, lstm_hidden, fc1_out, fc2_in, fc2_out, fc3_out, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden, fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self.fc3 = nn.Linear(fc2_out, fc3_out)
        self.output_layer = nn.Linear(fc3_out, output_dim)

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        h = F.relu(self.fc1(out[:, -1, :]))
        # Concatenate with zeros to match fc2's input size if needed
        if h.shape[1] < self.fc2.in_features:
            pad = torch.zeros(h.shape[0], self.fc2.in_features - h.shape[1], device=h.device)
            h = torch.cat([h, pad], dim=1)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        out = self.output_layer(h)
        return out, hx

class ActorCritic(nn.Module):
    def __init__(self, input_dim=5, window_size=16, lstm_hidden=128):
        super().__init__()
        # Actor: output_dim=3, Critic: output_dim=1
        self.actor = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 3)
        self.critic = LSTM_MLP(input_dim, lstm_hidden, 32, 80, 64, 64, 1)

    def forward(self, x, actor_hx=None, critic_hx=None):
        actor_out, actor_hx = self.actor(x, actor_hx)
        critic_out, critic_hx = self.critic(x, critic_hx)
        return actor_out, critic_out, actor_hx, critic_hx


