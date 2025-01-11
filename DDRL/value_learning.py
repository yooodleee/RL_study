"""
Functions for state value and action-value learning.

Value functions estimate the expected return (discounted sum of rewards) that
    can be collected by an agent under a given policy of behavior. This subpackage
    implements a number of functions for value learning in discrete scalar action
    spaces.
Actions are assumed to be represented as indices in the range 
    `[0, A)` where `A` is the number of distinct actions.
"""

from typing import NamedTuple, Optional
import torch
import torch.nn.functional as F

from . import base
from . import multistep


class QExtra(NamedTuple):
    target: Optional[torch.Tensor]
    td_error: Optional[torch.Tensor]


