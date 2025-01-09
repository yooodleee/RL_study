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


class DuelingDqnMlpNet(nn.Module):
    """
    MLP Dueling DQN network.
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
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> DqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        """

        features = self.body(x)
        advantages = self.advantage_head(features)  # [batch_size, action_dim]
        values = self.value_head(features)  # [batch_size, 1]
        q_values = values + (
            advantages - torch.mean(advantages, dim=1, keepdim=True)
        )   # [batch_size, action_dim]

        return DqnNetworkOutputs(q_values=q_values)
    

class C51DqnMlpNet(nn.Module):
    """
    C51 DQN MLP network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atoms: torch.Tensor,
    ):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            atoms: the support for q value distribution, used here to turn Z
                into Q values
        """
        if action_dim < 1:
            raise ValueError(
                f'Expect action_dim to be a positive integer, got {action_dim}'
            )
        if state_dim < 1:
            raise ValueError(
                f'Expect state_dim to be a positive integer, got {state_dim}'
            )
        if len(atoms.shape) != 1:
            raise ValueError(
                f'Expect atoms to be a 1D tensor, got {atoms.shape}'
            )
        
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * self.num_atoms),
        )
    
    def forwarad(
        self, x: torch.Tensor
    )-> C51NetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        """
        x = self.body(x)

        q_logits = x.view(
            -1, self.action_dim, self.num_atoms
        )   # [batch_size, action_dim, num_atoms]
        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)    # [batch_size, action_dim]

        return C51NetworkOutputs(
            q_logits=q_logits, q_values=q_values
        )


class RainbowDqnMlpNet(nn.Module):
    """
    Rainbow combines C51, dueling architecture, and noisy net.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atoms: torch.Tensor,
    ):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear layer
            atoms: the support for q value distribution, used here to turn
                Z into Q values
        """
        if action_dim < 1:
            raise ValueError(
                f'Expect action_dim to be a positive integer, got {action_dim}'
            )
        if state_dim < 1:
            raise ValueError(
                f'Expect state_dim to be a positive integer, got {state_dim}'
            )
        if len(atoms.shape) != 1:
            raise ValueError(
                f'Expect atoms to be a 1D tensor, got {atoms.shape}'
            )
        
        super().__init__()

        self.action_dim = action_dim
        self.atoms = atoms
        self.num_atoms = atoms.size(0)

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.advantage_head = nn.Sequential(
            common.NoisyLinear(128, 128),
            nn.ReLU(),
            common.NoisyLinear(128, action_dim * self.num_atoms),
        )
        self.value_head = nn.Sequential(
            common.NoisyLinear(128, 128),
            nn.ReLU(),
            common.NoisyLinear(128, 1 & self.num_atoms),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> C51NetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        """
        x = self.body(x)
        advantages = self.advantage_head(x)
        values = self.value_head(x)

        advantages = advantages.view(
            -1, self.action_dim, self.num_atoms
        )
        values = values.view(-1, 1, self.num_atoms)

        q_logits = values + (
            advantages - torch.mean(advantages, dim=1, keepdim=True)
        )
        q_logits = q_logits.view(
            -1, self.action_dim, self.num_atoms
        )   # [batch_size, action_dim, num_atoms]

        q_dist = F.softmax(q_logits, dim=-1)
        atoms = self.atoms[None, None, :].to(device=x.device)
        q_values = torch.sum(q_dist * atoms, dim=-1)

        return C51NetworkOutputs(
            q_logits=q_logits, q_values=q_values
        )
    
    def reset_noise(self)-> None:
        """
        Reset noisy layer.
        """
        # Combine two lists into one: list(chain(*zip(a, b)))
        for module in list(chain(*zip(
            self.advantage_head.modules(), self.value_head.modules()
        ))):
            if isinstance(module, common.NoisyLinear):
                module.reset_noise()


class QRDqnMlNet(nn.Module):
    """
    Quantile Regression DQN MLP network.
    """

    def __init__(
        self,
        state_dim: int, 
        action_dim: int,
        quantiles: torch.Tensor,
    ):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network
            action_dim: the number of units for the output linear shape
            quantile: the quantile for QR DQN
        """
        if action_dim < 1:
            raise ValueError(
                f'Expect action_dim to be a positive integer, got {action_dim}'
            )
        if state_dim < 1:
            raise ValueError(
                f'Expect state_dim to be a positive integer, got {state_dim}'
            )
        if len(quantiles.shape) != 1:
            raise ValueError(
                f'Expect quantiles to be a 1D tensor, got {quantiles.shape}'
            )
        
        super().__init__()
        self.taus = quantiles
        self.num_taus = quantiles.size(0)
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * self.num_taus),
        )
    
    def forward(
        self, x: torch.Tensor
    )-> QRDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        """
        # No softmax as the model is trying to approximate the 'whole' probability 
        # distributions
        q_dist = self.body(x).view(
            -1, self.num_taus, self.action_dim
        )   # [batch_size, num_taus, action_dim]
        q_values = torch.mean(q_dist, dim=1)

        return QRDqnNetworkOutputs(
            q_values=q_values, q_dist=q_dist
        )


