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


