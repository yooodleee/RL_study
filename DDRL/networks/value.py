"""
Networks for value-based learning methods 
    like DQN and it's varianets.
"""

from itertools import chain
from typing import (
    NamedTuple, Optional, Tuple
)
import torch
from torch import nn
import torch.nn.functional as F

# pylint: disable=import-error
from networks import common


class DqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor


class C51NetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_logits: torch.Tensor # Use logits and log_softmax() when calculate loss to avoid log() on zero cause NaN


class QRDqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_dist: torch.Tensor


class IqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    q_dist: torch.Tensor
    taus: torch.Tensor


class RnnDqnNetworkInputs(NamedTuple):
    s_t: torch.Tensor
    a_tm1: torch.Tensor
    r_t: torch.Tensor   # reward for (s_tm1, a_tm1), but received at current timestep.
    hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]


class RnnDqnNetworkOutputs(NamedTuple):
    q_values: torch.Tensor
    hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]


class NguNetworkInputs(NamedTuple):
    """
    Never give up agent network input.
    """

    s_t: torch.Tensor
    a_tm1: torch.Tensor
    ext_r_t: torch.Tensor   # extrinsic reward for (s_tm1, a_tm1), but received at current timestep.
    int_r_t: torch.Tensor   # intrinsic reward for (s_tm1)
    policy_index: torch.Tensor  # index for intrinsic reward scale beta and discount
    hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]


class Agent57NetworkInputs(NamedTuple):
    """
    Never give up agent network input.
    """

    s_t: torch.Tensor
    a_tm1: torch.Tensor
    ext_r_t: torch.Tensor   # extrinsic reward for (s_tm1, a_tm1), but received at current timestep.
    int_r_t: torch.Tensor   # intrinsic reward for (s_tm1)
    policy_index: torch.Tensor  # index for intrinsic reward scale beta and discount
    ext_hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]
    int_hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]


class Agent57NetworkOutputs(NamedTuple):
    ext_q_values: torch.Tensor
    int_q_values: torch.Tensor
    ext_hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]
    int_hidden_s: Optional[
        Tuple[torch.Tensor, torch.Tensor]
    ]


# =============================================
# Fully connected Neural Networks
# =============================================


class DqnMlpNet(nn.Module):
    """
    MLP DQN network.
    """

    def __init__(
        self, state_dim: int, action_dim: int
    ):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
        """
        if action_dim < 1:
            raise ValueError(
                f'Expect action_dim to be a positive integer, got {action_dim}'
            )
        if state_dim < 1:
            raise ValueError(
                f'Expect state_dim to be a positive integer, got {state_dim}'
            )
        
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> DqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        """

        q_values = self.body(x) # [batch_size, action_dim]
        return DqnNetworkOutputs(q_values=q_values)


