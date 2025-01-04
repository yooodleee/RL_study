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