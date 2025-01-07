"""
Common components for network.
"""

from typing import Tuple
import math
import torch
from torch import nn
from torch.nn import functional as F


def calc_conv2d_output(
    h_w: Tuple,
    kernel_size: int = 1,
    stride: int = 1,
    pad: int = 0,
    dilation: int = 1,
)-> Tuple[int, int]:
    """
    Takes a tuple of (h, w) and returns a tuple of (h, w).
    """

    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)
    
    h = math.floor(
        ((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) -1) / stride) + 1
    )
    w = math.floor(
        ((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1
    )
    
    return (h, w)

def initialize_weights(net: nn.Module)-> None:
    """
    Initialize weights for Conv2d and Linear layer using kaming initializer.
    """
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(
            module, (nn.Conv2d, nn.Linear)
        ):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


