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


