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
    

