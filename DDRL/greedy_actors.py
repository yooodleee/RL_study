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


