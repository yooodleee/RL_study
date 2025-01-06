"""
Agent57 agent class.

From the paper "Agent57: Outperforming the Atari Human Benchmark.
    https://arxiv.org/pdf/2003.13350.
"""

from typing import (
    Iterable,
    Mapping,
    Optional,
    Tuple,
    NamedTuple,
    Text,
)
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import replay as replay_lib
from .. import types as types_lib
import normalizer
import transforms
import nonlinear_bellman
import base
import distributed
import bandit
from curiosity import EpisodicBounusModule, RndLifeLongBonusModule
from networks.value import Agent57NetworkInputs

torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class Agent57Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env, step().

    last_action is the last agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]   # q values for s_t, computed from both ext_q_network and int_q_network
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    ext_r_t: Optional[float]    # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]    # intrinsic reward for (s_tm1)
    policy_index: Optional[int] # intrinsic reward scale beta index
    beta: Optional[float]   # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    ext_init_h: Optional[np.ndarray]    # nn.LSTM initial hidden state, from ext_q_network
    ext_init_c: Optional[np.ndarray]    # nn.LSTM initial cell state, from ext_q_network
    int_init_h: Optional[np.ndarray]    # nn.LSTM initial hidden state, from in_q_network
    int_init_c: Optional[np.ndarray]    # nn.LSTM initial cell state, from int_q_network


TransitionStructure = Agent57Transition(
    s_t=None,
    a_t=None,
    q_t=None,
    prob_a_t=None,
    last_action=None,
    ext_r_t=None,
    int_r_t=None,
    policy_index=None,
    beta=None,
    discount=None,
    ext_init_h=None,
    ext_init_c=None,
    int_init_h=None,
    int_init_c=None,
)


def compute_transformed_q(
    ext_q: torch.Tensor,
    int_q: torch.Tensor,
    beta: torch.Tensor,
)-> torch.Tensor:
    """
    Returns transformed state-action values from ext_q and int_q.
    """
    if not isinstance(beta, torch.Tensor):
        beta = torch.tensor(beta).expand_as(
            int_q
        ).to(device=ext_q.device)
    
    if len(beta.shape) < len(int_q.shape):
        beta = beta[..., None].expand_as(int_q)
    
    return transforms.signed_hyperbolic(
        transforms.signed_parabolic(ext_q) \
        + beta * transforms.signed_parabolic(int_q)
    )


def no_autograd(net: torch.nn.Module):
    """
    Disable autograd for a network.
    """

    for p in net.parameters():
        p.requires_grad = False


class Agent(types_lib.Agent):
    """
    Agent57 actor.
    """

    def __init__(
        self,
        rank: int,
        data_queue: multiprocessing.Queue,
        neetwork: torch.nn.Module,
        rnd_target_network: torch.nn.Module,
        rnd_predictor_network: torch.nn.Module,
        embedding_network: torch.nn.Module,
        random_state: np.random.RandomState,
        ext_discount: float,
        int_discount: float,
        num_actors: int,
        action_dim: int,
        unroll_length: int,
        burn_in: int,
        num_policies: int,
        policy_beta: float,
        ucb_window_size: int,
        ucb_beta: float,
        ucb_epsilon: float,
        episodic_memory_capacity: int,
        reset_episodic_memory: bool,
        num_neighbors: int,
        cluster_distance: float,
        kernel_epsilon: float,
        max_similarity: float,
        actor_update_interval: int,
        device: torch.device,
        shared_params: dict,
    )-> None:
        """
        Args:
            rank: the rank number for the actor.
            data_queue: a multiprocessing.Queue to send collected transitions to
                learner process.
            network: the Q network for actor to make action choice.
            rnd_target_network: RND random target network.
            rnd_predictor_network: RND predictor target network.
            embedding_network: NGU action prediction network.
            random_state: random state.
            ext_discount: extrinsic reward discount.
            int_discount: intrinsic reward discount.
            num_actors: number of actors.
            action_dim: number of valid actions in the environment.
            unroll_length: how many agent time step to unroll transitions before
                put on to queue.
            burn_in: two consecutive unrolls will overlap on burn_in+1 steps.
            num_policies: number of exploring and exploiting policies.
            policy_beta: intrinsic reward scale beta.
            ucb_window_size: window size of the sliding window UCB algorithm.
            ucb_beta: beta for the sliding window UCB algorithm.
            ucb_epsilon: exploration epsilon for sliding window UCB algorithm.
            episodic_memory_capacity: maximum capacity of episodic memory.
            reset_episodic_memory: Reset the episodic_memory on envery episode.
            num_neighbors: number of K-NN neighbors for compute episodic bouns.
            cluster_distance: K-NN neighbors cluster distance for compute episodic
                bonus.
            kernel_epislon: K-NN kernel epsilon for compute episodic bonus.
            max_similarity: maximum similarity for compute episodic bonus.
            actor_update_interval: the frequency to update actor's Q network.
            device: PyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters
                for actors.
        """
        if not 0.0 <= ext_discount <= 1.0:
            raise ValueError(
                f'Expect ext_discount t0 be [0.0, 1.0), got {ext_discount}'
            )
        if not 0.0 <= int_discount <= 1.0:
            raise ValueError(
                f'Expect int_discount to be [0.0, 1.0), got {int_discount}'
            )
        if not 0 < num_actors:
            raise ValueError(
                f'Expect num_actors to be positive integer, got {num_actors}'
            )
        if not 0 < action_dim:
            raise ValueError(
                f'Expect action_dim to be positive integer, got {action_dim}'
            )
        if not 1 <= unroll_length:
            raise ValueError(
                f'Expect unroll_length to be integer greater than or equal to 1, got {unroll_length}'
            )
        if not 0 <= burn_in < unroll_length:
            raise ValueError(
                f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}'
            )
        if not 1 <= num_policies:
            raise ValueError(
                f'Expect num_policies to be integer greater than or equal to 1, got {num_policies}'
            )
        if not 0.0 <= policy_beta <= 1.0:
            raise ValueError(
                f'Expect policy_beta to be [0.0, 1.0), got {policy_beta}'
            )
        if not 1 <= ucb_window_size:
            raise ValueError(
                f'Expect ucb_window_size to be integer greater than or equal to 1, got {ucb_window_size}'
            )
        if not 0.0 <= ucb_beta <= 100.0:
            raise ValueError(
                f'Expect ucb_beta to be [0.0, 100.0), got {ucb_beta}'
            )
        if not 0.0 <= ucb_epsilon <= 1.0:
            raise ValueError(
                f'Expect ucb_epsilon to be [0.0, 1.0), got {ucb_epsilon}'
            )
        if not 1 <= episodic_memory_capacity:
            raise ValueError(
                f'Expect episodic_memory_capacity to be integer greater than or equal to 1, got {episodic_memory_capacity}'
            )
        if not 1 <= num_neighbors:
            raise ValueError(
                f'Expect num_neighbors to be integer greater than or equal to 1, got {num_neighbors}'
            )
        if not 0.0 <= cluster_distance:
            raise ValueError(
                f'Expect cluster_disctance to be [0.0, 1.0], got {cluster_distance}'
            )
        if not 0.0 <= kernel_epsilon <= 1.0:
            raise ValueError(
                f'Expect kerenl_epsilon to be [0.0, 1.0], got {kernel_epsilon}'
            )
        if not 1 <= actor_update_interval:
            raise ValueError(
                f'Expect actor_update_interval to be integer greater than or equal to 1, got {actor_update_interval}'
            )
        
        self.rank = rank
        self.agent_name = f'Agent57-actor{rank}'

        self._network = neetwork.to(device=device)
        self._rnd_target_network = rnd_target_network.to(device=device)
        self._rnd_predictor_network = rnd_predictor_network.to(device=device)
        self._embedding_network = embedding_network.to(device=device)

        # Disable autograd for actor's Q networks, embedding, and RND networks.
        no_autograd(self._network)
        no_autograd(self._rnd_target_network)
        no_autograd(self._rnd_predictor_network)
        no_autograd(self._embedding_network)

        self._shared_params = shared_params

        self._queue = data_queue
        self._device = device
        self._random_state = random_state
        self._num_actors = num_actors
        self._action_dim = action_dim
        self._actor_update_interval = actor_update_interval
        self._num_policies = num_policies

        self._unroll = replay_lib.Unroll(
            unroll_length=unroll_length,
            overlap=burn_in + 1,    # Plus 1 to add room for shift during learning
            structure=TransitionStructure,
            cross_episode=False,
        )

        # Meta-collector
        self._meta_coll = bandit.SimplifiedSlidingWindowUCB(
            self._num_policies,
            ucb_window_size,
            self._random_state,
            ucb_beta,
            ucb_epsilon,
        )

        self._betas, self._gammas = distributed.get_ngu_polic_betas_and_discounts(
            num_policies=num_policies,
            beta=policy_beta,
            gamma_max=ext_discount,
            gamma_min=int_discount,
        )
        self._policy_index = None
        self._policy_beta = None
        self._policy_discount = None
        self._sample_policy()

        self._reset_episodic_memory = reset_episodic_memory

        # E-greedy policy epsilon, rank 0 has the lowest noise, while rank N-1
        # has the highest noise.
        epsilons = distributed.get_actor_exploration_epsilon(num_actors)
        self._exploration_epsilon = epsilons[self.rank]

        # Episodic intrinsic bonus module
        self._episodic_module = EpisodicBounusModule(
            embedding_network=self._embedding_network,
            device=device,
            capacity=episodic_memory_capacity,
            num_neighbors=num_neighbors,
            kernel_epsilon=kernel_epsilon,
            cluster_distance=cluster_distance,
            max_similarity=max_similarity,
        )

        # Lifelong intrinsic bonus module
        self._lifelong_module = RndLifeLongBonusModule(
            target_network=self._rnd_target_network,
            predictor_network=self._rnd_predictor_network,
            device=device,
            discount=int_discount,
        )

        self._episodic_returns = 0.0
        self._last_action = None
        self._episodic_bonus_t = None
        self._lifelong_bonus_t = None
        self._ext_lstm_state = None # Stores nn.LSTM hidden state and cell state. for extrinsic Q network
        self._int_lstm_state = None # Stores nn.LSTM hidden state and cell state. for intrinsic Q network

        self._step_t = -1
    
    