"""
Networks for policy-based learning methods like Actor-Critic
    and it's variants
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import (
    NamedTuple, Optional, Tuple
)

# pylint: disable=import-error
from networks import common


class ActorNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor


class CriticNetworkOutputs(NamedTuple):
    value: torch.Tensor


class ActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    value: torch.Tensor


class ImpalaActorCriticNetworkOutputs(NamedTuple):
    pi_logits: torch.Tensor
    value: torch.Tensor
    hidden_s: torch.Tensor


class ImpalaActorCriticNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor   # reward for (s_tm1, a_tm1), but received at current timestep.
    done: torch.Tensor
    hidden_s: Optional[Tuple[torch.Tensor]]


class RndActorCriticNetworkOutputs(NamedTuple):
    """
    Random Network Distillation.
    """

    pi_logits: torch.Tensor
    int_baseline: torch.Tensor  # intrinsic value head
    ext_baseline: torch.Tensor  # extrinsic value head


# =====================================================
# Fully connected Neural Networks
# =====================================================


class ActorMlpNet(nn.Module):
    """
    Actor MLP network.
    """

    def __init__(
        self, state_dim: int, action_dim: int
    )-> None:
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> ActorNetworkOutputs:
        """
        Given raw state x, predict the action probability distribution.
        """
        # Predict action distribution wrt policy
        pi_logits = self.net(x)

        return ActorNetworkOutputs(
            pi_logits=pi_logits
        )


class CriticMlpNet(nn.Module):
    """
    Critic MLP network.
    """

    def __init__(
        self, state_dim: int
    )-> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> CriticNetworkOutputs:
        """
        Given raw state x, predict the state-value.
        """
        value = self.net(x)
        return CriticNetworkOutputs(value=value)
    

class ActorCriticMlpNet(nn.Module):
    """
    Actor-Critic MLP network.
    """

    def __init__(
        self, state_dim: int, action_dim: int
    )-> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.baseline_head = nn.Sequential(
            # nn.Linear(64, 64),
            # nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> ActorCriticNetworkOutputs:
        """
        Given raw state x, predict the action probability distribution
            and state-values.
        """
        # Extract features from raw input state.
        features = self.body(x)

        # Predict action distribution wrt policy
        pi_logits = self.policy_head(features)

        # Predict state-value
        value = self.baseline_head(features)

        return ActorCriticNetworkOutputs(
            pi_logits=pi_logits, value=value
        )


class GaussianActorMlpNet(nn.Module):
    """
    Gaussian Actor MLP network for continuous action space.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
    )-> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mu_head = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
        )

        # self.sigma_head = nn.Sequential(
        #       nn.Linear(hidden_size, hidden_size)
        #       nn.Tanh(),
        #       nn.Linear(hidden_size, action_dim),
        # )
        self.logstd = nn.Parameter(
            torch.zeros(1, action_dim)
        )
    
    def forward(
        self, x: torch.Tensor
    )-> Tuple[torch.Tensor]:
        """
        Given raw state x, predict the action probability distribution
            and state-value.
        """
        features = self.body(x)

        # Predict action distributions wrt policy
        pi_mu = self.mu_head(features)
        # pi_sigma = torch.exp(self.sigma_head(features))

        logstd = self.logstd.expand_as(pi_mu)
        pi_sigma = torch.exp(logstd)

        return pi_mu, pi_sigma


class GaussianCriticMlpNet(nn.Module):
    """
    Gaussian Critic MLP network for continuous action space.
    """

    def __init__(
        self, state_dim: int, hidden_size: int
    )-> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> torch.Tensor:
        """
        Given raw state x, predict the state-value.
        """

        # Predict state-value
        value = self.net(x)

        return value


