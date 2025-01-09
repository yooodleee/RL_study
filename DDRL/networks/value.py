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
    q_values = torch.Tensor


class C51NetworkOutputs(NamedTuple):
    q_values = torch.Tensor
    q_logits = torch.Tensor # Use logits and log_softmax() when calculate loss to avoid log() on zero cause NaN


class QRDqnNetworkOutputs(NamedTuple):
    q_values = torch.Tensor
    q_dist = torch.Tensor


