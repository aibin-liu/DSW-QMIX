import torch as th
import torch.nn as nn
import torch.nn.functional as F


class UrgencyHead(nn.Sequential):
    """
    State-dependent cost weight lambda_p(s) for DSW-QMIX.
    Maps the global state to a strictly positive scalar that scales the cost mixer Q_p.
    """

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: th.Tensor) -> th.Tensor:
        """(N, state_dim) -> (N, 1), values > 0 (softplus + floor)."""
        return F.softplus(super().forward(global_state)).clamp(min=1e-6)
