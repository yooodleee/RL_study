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


class IqnMlpNet(nn.Module):
    """
    Implicity Quantiel MLP network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int,
    ):
        """
        Args:
            state_dim: the shape of the inpute tensor to the nueral network
            action_dim: the number of units for the output linear layer
            latent_dim: the cos embedding linear layer input shapes
        """
        if action_dim < 1:
            raise ValueError(
                f'Expect action_dim to be a positive integer, got {action_dim}'
            )
        if state_dim < 1:
            raise ValueError(
                f'Expect state_dim to be a positive integer, got {state_dim}'
            )
        if latent_dim < 1:
            raise ValueError(
                f'Expect latent_dim to be a positieve integer, got {latent_dim}'
            )
        
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.pis = torch.arange(
            1, self.latent_dim + 1
        ).float() * 3.141592653589793  # [latent_dim]

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.embedding_layer = nn.Linear(latent_dim, 128)
        self.value_head = nn.Linear(128, action_dim)

    def sample_tuas(
        self, batch_size: int, num_taus: int
    )-> torch.Tensor:
        """
        Returns sampled batch taus.
        """
        taus = torch.rand(
            (batch_size, num_taus)
        ).to(dtype=torch.float32)

        assert taus.shape == (batch_size, num_taus)
        return taus
    
    def forward(
        self, x: torch.tensor, num_taus: int = 32
    )-> IqnNetworkOutputs:
        """
        Args:
            state: environment state, shape (B, state_shape)
            taus: tau embedding samples, shape (B, num_taus)

        Returns:
            q_values: # [batch_size, action_dim]
            q_dist: # [batch_size, action_dim, num_taus]
        """
        batch_size = x.shape[0]
        # Apply DQN to embed state.
        features = self.body(x)

        taus = self.sample_tuas(
            batch_size, num_taus
        ).to(device=x.device)

        # Embed taus with cosine embedding + linear layer.
        # cos(pi * i * tua) for i = 1,...,latents for each batch_element x sample.
        # Broadcast everything to batch x num_taus x latent_dim.
        pis = self.pis[None, None, :].to(device=x.device)
        tau_embedding = torch.cos(pis * taus[:, :, None])   # [batch_size, num_tuas, latent_dim]

        # Merge batch and taus dimension before input to embedding layer.
        tau_embedding = tau_embedding.view(
            batch_size * num_taus, -1
        )   # [batch_size x num_taus, latent_dim]
        tau_embedding = F.relu(
            self.embedding_layer(tau_embedding)
        )   # [batch_size x num_taus, embedding_layerlayer]

        # Reshape/broadcast both embeddings to batch x num_taus x state_dim
        # and multiply together, before applying value head.
        tau_embedding = tau_embedding.view(
            batch_size, num_taus, -1
        )
        head_input = tau_embedding * features[:, None, :]   # [batch_size, num_taus, embedding_layerlayer_output]

        # Merge head input dimensions.
        head_input = head_input.view(
            -1, self.embedding_layer.out_features
        )

        # No softmax as the model is trying to approximate the 'whole' probability distributions
        q_dist = self.value_head(head_input)    # [batch_size x num_taus, action_dim]
        q_dist = q_dist.view(
            batch_size, -1, self.action_dim
        )   # [batch_size, num_taus, action_dim]
        q_values = torch.mean(q_dist, dim=1)    # [batch_size, action_dim]

        return IqnNetworkOutputs(
            q_values=q_values, q_dist=q_dist, taus=taus
        )
    

class DrqnMlpNet(nn.Module):
    """
    DRQN MLP network.
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
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )
        self.value_head = nn.Linear(
            self.lstm.hidden_size, action_dim
        )

    def forward(
        self, x: torch.Tensor, hidden_s: None
    )-> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is batch (B) of length (T) states.
        B refers to the batch size T refert to the time dimension.

        Args:
            x: the batch of state tensor, shape [B, T, state_shape]
            hidden_s: the initial/last time step hidden state from lstm
        
        Returns:
            q_values: state-action values
            hidden_s: hidden state from LSTM layer
        """
        # Expect x shape to be [B, T, state_shape]
        assert len(x.shape) == 3
        B = x.shape[0]
        T = x.shape[1]

        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        x = self.body(x)
        x = x.view(B, T, -1)    # LSTM expect rank 3

        x, hidden_s = self.lstm(x, hidden_s)

        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        q_values = self.value_head(x)
        q_values = q_values.view(B, T, -1)  # reshape to in the range [B, T, action_dim]

        return RnnDqnNetworkOutputs(
            q_values=q_values, hidden_s=hidden_s
        )
    
    def get_initial_hidden_state(
        self, batch_size: int
    )-> Tuple[torch.Tensor]:
        """
        Get initial LSTM hidden state, which is all zeros,
            should call at the beginning of new episode, or every training batch.
        """
        # Shape should be num_layers, batch_size, hidden_size, 
        # note lstm expect two hidden states.
        return tuple(
            torch.zeros(
                self.lstm.num_layers,
                batch_size,
                self.lstm.hidden_size,
            ) for _ in range(2)
        )


class R2d2DqnMlpNet(nn.Module):
    """
    R2D2 DQN MLP network.
    """

    def __init__(
        self, state_dim: int, action_dim: int
    ):
        """
        Args:
            state_dim: the shape of the inpute tensor to the neural network
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
        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Feature representation output size + oen-hot of last action + last reward.
        core_output_size = 128 + self.action_dim + 1

        self.lstm = nn.LSTM(
            input_size=core_output_size,
            hidden_size=128,
            num_layers=1
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self, input_: RnnDqnNetworkInputs
    )-> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refert to the time dimension.

        Args:
            x: the RnnDqnNetworkInputs object which contains the following attributes:
                s_t: timestep t environment state, shape [T, B, state_shape].
                a_tim1: action taken in t-1 timestep, shape [T, B].
                r_t: reward for state-action pair (s_tm1, a_tm1), shape [T, B].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tenwors
                    each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object which the following attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output, tuple of two tensors each shape
                    (num_lstm_layer, B, lstm_hidden_size).
        """
        # Expect X shape to be [T, B, state_shape]
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        r_t = input_.r_t
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape    # [T, B, state_shape]
        x = torch.flatten(s_t, 0, 1)    # Merge batch and time dimension.

        x = self.body(x)
        x = x.view(T * B, -1)

        # Append rewrad and one hot last action.
        one_hot_a_tm1 = F.one_hot(
            a_tm1.view(T * B), self.action_dim
        ).float().to(device=x.device)

        reward = r_t.view(T * B, 1)
        core_input = torch.cat(
            [x, reward, one_hot_a_tm1],
            dim=-1
        )
        core_input = core_input.view(T, B, -1)  # LSTM expect rank 3 tensor.

        # If no hidden_s provided, use zero start strategy
        if hidden_s is None:
            hidden_s = self.get_initial_hidden_state(batch_size=B)
            hidden_s = tuple(
                s.to(device=x.device) for s in hidden_s
            )
        
        x, hidden_s = self.lstm(core_input, hidden_s)

        x = torch.flatten(x, 0, 1)  # Merge batch and time dimension.
        advantages = self.advantage_head(x) # [T*B, action_dim]
        values = self.value_head(x) # [T*B, 1]

        q_values = values + (
            advantages - torch.mean(advantages, dim=1, keepdim=True)
        )
        q_values = q_values.view(T, B, -1)  # reshape to in the range [B, T, action_dim]

        return RnnDqnNetworkOutputs(
            q_values=q_values, hidden_s=hidden_s
        )
    
    def get_initial_hidden_state(
        self, batch_size: int
    )-> Tuple[torch.Tensor]:
        """
        Get initial LSTM hidden state, which is all zeros,
            should call at the beginning of new episode, or every training batch.
        """
        # Shape should be num_layers, batch_size, hidden_size, 
        # note lstm expects two hidden states.
        return tuple(
            torch.zeros(
                self.lstm.num_layers,
                batch_size,
                self.lstm.hidden_size,
            ) for _ in range(2)
        )


class NguDqnMlpNet(nn.Module):
    """
    NGU DQN MLP network.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_policies: int,
    ):
        """
        Args:
            state_dim: the shape of the input tensor to the neural network.
            action_dim: the number of units for the output linear layer.
            num_policies: the number of mixtures for intrinsic reward scale betas.
        """
        super().__init__()
        self.action_dim = action_dim
        self.num_policies = num_policies

        self.body = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # Core input includes:
        # feature reprentation output size.
        # one-hot of intrinsic reward scale beta
        # one-hot of last action
        # last intrinsic reward
        # last extrinsic reward
        core_output_size = 128 + self.num_policies + self.action_dim + 1 + 1

        self.lstm = nn.LSTM(
            input_size=core_output_size,
            hidden_size=128,
            num_layers=1,
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
    
    def forward(
        self, input_: NguNetworkInputs
    )-> RnnDqnNetworkOutputs:
        """
        Given state, return state-action value for all possible actions.
        where the state is a batch (B) of length (T) states.
        B refers to the batch size T refers to the time dimension.

        Args:
            x: the NguNetworkInputs object which contains the following attributes:
                s_t: timestep t envrionment state, shape [T, B, state_shape].
                a_tm1: action taken in s_tm1, shape [T, B].
                ext_r_t: extrinsic reward for state-action pair (s_tm1, a_tm1), shape [T, B].
                int_r_t: intrinsic reward for state s_tm1, shape [T, B].
                policy_index: the index for the pair of intrinsic reward scale beta and
                    discount gamma, shape [T, B].
                hidden_s: LSTM layer hidden state from t-1 timestep, tuple of two tensors 
                    each shape (num_lstm_layers, B, lstm_hidden_size).

        Returns:
            RnnDqnNetworkOutputs object with the fowllowing attributes:
                q_values: state-action values.
                hidden_s: hidden state from LSTM layer output.
        """
        # Expect x shape to be [T, B, state_shape]
        s_t = input_.s_t
        a_tm1 = input_.a_tm1
        ext_r_t = input_.ext_r_t
        int_r_t = input_.int_r_t
        policy_index = input_.policy_index
        hidden_s = input_.hidden_s

        T, B, *_ = s_t.shape    # [T, B, state_shape]
        