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


class NatureCnnBackboneNet(nn.Module):
    """
    DQN Nature paper conv2d layers backbone, returns feature representation
        vector.
    """

    def __init__(self, state_dim: tuple)-> None:
        super().__init__()

        # Compute the output shape of final conv2d layer
        c, h, w = calc_conv2d_output((h, w), 8, 4)
        h, w = calc_conv2d_output((h, w), 4, 2)
        h, w = calc_conv2d_output((h, w), 3, 1)

        self.out_features = 64 * h * w

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=c, out_channels=32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1
            ),
            nn.ReLU(),
            nn.Flatten(),
        )
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        Given raw state images, returns feature representation vector.
        """
        return self.net()
    


class ResnetBlock(nn.Module):
    """
    Basic 3x3 residual block.
    """

    def __init__(
        self,
        num_planes: int,
    )-> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                is_channels=num_planes,
                out_channels=num_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_planes),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_planes,
                out_channels=num_planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_planes),
        )
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)

        return out
    

class NoisyLinear(nn.Module):
    """
    Factorized NoisyLinear layer with bias.

    Code adapted form:
        https://github.com/kaixhin/Rainbow/blob/master/model.py
    """

    def __init__(
        self,
        in_features,
        out_features,
        std_init=0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )
        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )
        self.register_buffer(
            'bias_epsilon', torch.empty(out_features)
        )
        self.register_parameter()
        self.reset_noise()
    
    def reset_parameter(self):
        """
        Only call this during initialization.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """
        Should call this after doing backpropagation.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)