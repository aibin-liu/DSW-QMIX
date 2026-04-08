import torch

import torch.nn as nn
import torch.nn.functional as f


class RNNAgent(nn.Module):
    """
    RNN-based agent for multi-agent reinforcement learning.
    Processes observation sequences using an RNN to generate actions.
    """
    
    def __init__(self, input_shape, n_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, observations, hidden_state=None):
        """
        Args:
            observations: (batch, seq_len, input_size)
            hidden_state: (num_layers, batch, hidden_size) or None
        
        Returns:
            q_values: (batch, seq_len, output_size)
            hidden_state: (num_layers, batch, hidden_size)
        """
        x = f.relu(self.fc1(observations))
        h = self.rnn(x, hidden_state)
        q = self.fc2(h)
        return q, h
    
    def init_hidden(self, batch_size):
        """Initialize hidden state."""
        return self.fc1.weight.new_zeros(1, batch_size, self.rnn.hidden_size)