import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_size, window, hidden_size = 128, action_dim = 3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_shared = nn.Linear(hidden_size, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x, hx = None):
        # x : (batch, window, features)
        out, hx = self.lstm(x, hx)
        h = torch.relu(self.fc_shared(out[:, -1, :]))
        #policy = torch.softmax(self.actor(h), dim=-1)
        return self.actor(h), self.critic(h), hx
