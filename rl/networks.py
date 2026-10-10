"""Actor and critic MLPs for continuous control."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def mlp(in_dim: int, out_dim: int, hidden: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(
        layer_init(nn.Linear(in_dim, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, hidden)),
        nn.Tanh(),
        layer_init(nn.Linear(hidden, out_dim), std=out_std),
    )


class GaussianActor(nn.Module):
    """pi(a|s) = N(mu_theta(s), diag(exp(log_std))^2), with a learnable state-independent log_std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.mu = mlp(obs_dim, act_dim, hidden, out_std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mu(obs), self.log_std.exp())

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.dist(obs).log_prob(act).sum(-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.v = mlp(obs_dim, 1, hidden, out_std=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs).squeeze(-1)
