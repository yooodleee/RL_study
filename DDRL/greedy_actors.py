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


def apply_egreedy_policy(
    q_values: torch.Tensor,
    epsilon: float,
    random_state: np.random.RandomState,    # pylint: disable=no-member
)-> types_lib.Action:
    """
    Apply e-greedy policy.
    """
    action_dim = q_values.shape[-1]
    if random_state.rand() <= epsilon:
        a_t = random_state.randint(0, action_dim)
    else:
        a_t = q_values.argmax(-1).cpu().item()
    return a_t


class EpsilonGreedyActor(types_lib.Agent):
    """
    DQN e-greedy actor.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,    # pylint: disable=no-member
        device: torch.device,
        name: str = 'DQN-greedy',
    ):
        self.agent_name = name
        self._device = device
        self._network = network.to(device=device)
        self._exploration_epsilon = exploration_epsilon
        self._random_state = random_state

    def step(
        self, timestep: types_lib.TimeStep
    )-> types_lib.Action:
        """
        Given current timestep, return best action.
        """
        return self._select_action(timestep)
    
    def reset(self)-> None:
        """
        Resets the agent's episodic state such as frame stack and action repeat.
        This method should be called at the beginning of every episode.
        """
    
    @torch.no_grad()
    def _select_action(
        self, timestep: types_lib.Timestep
    )-> types_lib.Action:
        """
        Samples action from eps-greedy policy wrt Q-values at given state.
        """
        s_t = torch.tensor(
            timestep.observation[None, ...]
        ).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t).q_values
        
        return apply_egreedy_policy(
            q_t, self._exploration_epsilon, self._random_state
        )
    
    @property
    def statistics(self)-> Mapping[Text, float]:
        """
        Returns current agent statistics as a dictionary.
        """
        return {
            'exploration_epsilon': self._exploration_epsilon,
        }



class IqnEpsilonGreedyActor(EpsilonGreedyActor):
    """
    IQN e-greedy actor.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,    # pylint: disable=no-member
        device: torch.device,
        tau_samples: int,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'IQN-greedy',
        )
        self._tau_samples = tau_samples
    
    @torch.no_grad()
    def _select_action(
        self, timestep: types_lib.TimeStep
    )-> types_lib.Action:
        """
        Samples action from eps-greedy policy wrt Q-values at given state.
        """
        s_t = torch.tensor(
            timestep.observation[None, ...]
        ).to(device=self._device, dtype=torch.float32)
        q_t = self._network(s_t, self._tau_samples).q_values

        return apply_egreedy_policy(
            q_t, self._exploration_epsilon, self._random_state
        )
    

class DrqnEpsilonGreedyActor(EpsilonGreedyActor):
    """
    DRQN e-greedy actor.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,    # pylint: disable=no-member
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'DRQN-greedy',
        )
        self._lstm_state = None
    
    @torch.no_grad()
    def _select_action(
        self, timestep: types_lib.TimeStep
    )-> types_lib.Action:
        """
        Samples action from eps-greedy policy wrt Q-values at given state.
        """
        s_t = torch.tensor(
            timestep.observation[None, None, ...]
        ).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(s.to(device=self._device) for s in self._lstm_state)
        network_output = self._network(s_t, hidden_s)
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s
        
        return apply_egreedy_policy(
            q_t, self._exploration_epsilon, self._random_state
        )
    
    def reset(self)-> None:
        """
        Reset hidden state to zeros at new episodes.
        """
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


class R2d2EpsilonGreedyActor(EpsilonGreedyActor):
    """
    R2D2 e-greedy actor.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        exploration_epsilon: float,
        random_state: np.random.RandomState,    # pylint: disable=no-member
        device: torch.device,
    ):
        super().__init__(
            network,
            exploration_epsilon,
            random_state,
            device,
            'R2D2-greedy',
        )
        self._last_action = None
        self._lstm_state = None
    
    @torch.no_grad()
    def _select_action(
        self, timestep: types_lib.TimeStep
    )-> types_lib.Action:
        """
        Samples action from eps-greedy policy wrt Q-values at given state.
        """
        s_t = torch.tensor(
            timestep.observation[None, ...]
        ).to(device=self._device, dtype=torch.float32)
        a_tm1 = torch.tensor(
            self._last_action
        ).to(device=self._device, dtype=torch.float64)
        r_t = torch.tensor(
            timestep.reward
        ).to(device=self._device, dtype=torch.float32)
        hidden_s = tuple(
            s.to(device=self._device) for s in self._lstm_state
        )

        network_output = self._network(
            RnnDqnNetworkInputs(
                s_t=s_t[None, ...],
                a_tm1=a_tm1[None, ...],
                r_t=r_t[None, ...],
                hidden_s=hidden_s[None, ...],
            )
        )
        q_t = network_output.q_values
        self._lstm_state = network_output.hidden_s

        a_t = apply_egreedy_policy(
            q_t, self._exploration_epsilon, self._random_state
        )
        self._last_action = a_t
        
        return a_t
    
    def reset(self)-> None:
        """
        Reset hidden state to zeros at new episodes.
        """
        self._last_action = 0   # During the first step of a new episode, use 'fake' previous action from network pass
        self._lstm_state = self._network.get_initial_hidden_state(batch_size=1)


