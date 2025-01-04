"""
Greedy actors for testing and evaluation.
"""

from typing import Mapping, Tuple, Text
import numpy as np
import torch

# pylint: disable=import-error
import types as types_lib
import distributions
from networks.policy import ImpalaActorCriticNetworkInputs
from networks.value import (
    RnnDqnNetworkInputs,
    NguNetworkInputs,
    Agnet57NetworkInputs,
)
from curiosity import (
    EpisodicBonusModule, RndLifeLongBonusModule
)
from agent57.agent import compute_transformed_q


HiddenState = Tuple[torch.Tensor, torch.Tensor]