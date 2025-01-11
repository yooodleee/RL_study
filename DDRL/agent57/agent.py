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
from curiosity import (
    EpisodicBounusModule, 
    RndLifeLongBonusModule
)

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
) -> torch.Tensor:
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
    ) -> None:
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
    
    @torch.no_grad()
    def step(
        self, timestep: types_lib.TimeStep
    ) -> types_lib.Action:
        """
        Given timestep, return action a_t, and push transition into global queue
        """
        self._step_t += 1
        self._episodic_returns += timestep.reward

        if self._step_t % self._actor_update_interval == 0:
            self._update_actor_network(False)
        
        q_t, a_t, prob_a_t, ext_hidden_s, int_hidden_s = self.act(timestep)

        transition = Agent57Transition(
            s_t=timestep.observation,
            a_t=a_t,
            q_t=q_t,
            prob_a_t=prob_a_t,
            last_action=self._last_action,
            ext_r_t=timestep.reward,
            int_r_t=self.intrinsic_reward,
            policy_index=self._policy_index,
            beta=self._policy_beta,
            discount=self._policy_discount,
            done=timestep.done,
            ext_init_h=self._ext_lstm_state[0].squeeze(1).cpu().numpy(),    # remove batch dimension
            ext_init_c=self._ext_lstm_state[1].squeeze(1).cpu().numpy(),
            int_init_h=self._int_lstm_state[0].squeeze(1).cpu().numpy(),    # remove batch dimension
            int_init_c=self._int_lstm_state[1].squeeze(1).cpu().numpy(),
        )

        unrolled_transition = self._unroll.add(
            transition, timestep.done
        )

        s_t = torch.from_numpy(
            timestep.observation[None, ...]
        ).to(device=self._device, dtype=torch.float32)

        # Compute lifelong intrinsic bonus
        self._lifelong_bonus_t = self._lifelong_module.compute_bonus(s_t)

        # Compute episodic intrinsic bonus
        self._episodic_bonus_t = self._episodic_module.compute_bonus(s_t)

        # Update local state
        self._last_action, self._ext_lstm_state, self._int_lstm_state = \
            a_t, ext_hidden_s, int_hidden_s
        
        if unrolled_transition is not None:
            self._put_unroll_onto_queue(unrolled_transition)

        return a_t
    
    def reset(self) -> None:
        """
        This method should be called at the beginning of every episode before
            take any action.
        """
        self._unroll.reset()

        if self._reset_episodic_memory:
            self._episodic_module.reset()
        
        self._update_actor_network(True)

        # Update Sliding Window UCB statistics.
        self._meta_coll.update(
            self._policy_index, self._episodic_returns
        )

        self._episodic_returns = 0.0

        # Agent57 actor samples a policy using the Sliding Window UCB algorithm,
        # then play a single episode.
        self._sample_policy()

        # During the first step of a new episode,
        # use 'fake' previous action and 'intrinsic' reward for network pass
        self._last_action = self._random_state.randint(
            0, self._action_dim
        )   # Initialize a_tm1 randomly
        self._episodic_bonus_t = 0.0
        self._lifelong_bonus_t = 0.0
        self._ext_lstm_state, self._int_lstm_state = self._network.get_initial_hidden_state(
            batch_size=1
        )
    
    def act(
        self,
        timestep: types_lib.TimeStep,
    ) -> Tuple[
        np.ndarray,
        types_lib.Action,
        float,
        HiddenState,
        HiddenState,
    ]:
        """
        Given state s_t and done marks, return an action.
        """
        return self._choose_action(timestep)
    
    @torch.no_grad()
    def _choose_action(
        self, timestep: types_lib.TimeStep
    ) -> Tuple[
        np.ndarray,
        types_lib.Action,
        float,
        HiddenState,
        HiddenState,
    ]:
        """
        Given state s_t, choose action a_t.
        """
        input_ = self._prepare_network_input(timestep)

        output = self._network(input_)
        ext_q_t = output.ext_q_values.squeeze()
        int_q_t = output.int_q_values.squeeze()

        q_t = compute_transformed_q(
            ext_q_t, int_q_t, self._policy_beta
        )

        a_t = torch.argmax(q_t, dim=-1).cpu().item()

        # Policy probability for a_t, the detailed equation is mentioned
        # in Agent57 paper.
        prob_a_t = 1 - (self._exploration_epsilon * ((self._action_dim - 1)\
                            / self._action_dim))
        
        # To make sure every actors generates the same amount of samples,
        # apply e-greedy after the network pass,
        # otherwise the actor with higher epsilons will generate more samples,
        # while the actor with lower epsilon will generate less samples.
        if self._random_state.rand() < self._exploration_epsilon:
            # randint() return random integers from low (inclusive) to high (exclusive).
            a_t = self._random_state.randint(
                0, self._action_dim
            )
            prob_a_t = self._exploration_epsilon / self._action_dim
        
        return (
            q_t.cpu().numpy(),
            a_t,
            prob_a_t,
            output.ext_hidden_s,
            output.int_hidden_s,
        )
    
    def _prepare_network_input(
        self, timestep: types_lib.TimeStep
    ) -> Agent57NetworkInputs:
        # Agent57 network expect input shape [T, B, state_shape],
        # and additionally 'last_action', 'extrinsic reward for last action',
        # last intrinsic reward, and intrinsic reward scale beta index.
        s_t = torch.from_numpy(
            timestep.observation[None, ...]
        ).to(device=self._device, dtype=torch.float32)
        last_action = torch.tensor(
            self._last_action,
            device=self._device,
            dtype=torch.int64,
        )
        ext_r_t = torch.tensor(
            timestep.reward,
            device=self._device,
            dtype=torch.float32,
        )
        int_r_t = torch.tensor(
            self.intrinsic_reward,
            device=self._device,
            dtype=torch.float32,
        )
        policy_index = torch.tensor(
            self._policy_index,
            device=self._device,
            dtype=torch.int64,
        )
        ext_hidden_s = Tuple(
            s.to(device=self._device) for s in self._ext_lstm_state
        )
        int_hidden_s = tuple(
            s.to(device=self._device) for s in self._int_lstm_state
        )

        return Agent57NetworkInputs(
            s_t=s_t.unsqueeze(0), # [T, B, state_shape]
            a_tm1=last_action.unsqueeze(0), # [T, B]
            ext_r_t=ext_r_t.unsqueeze(0),   # [T, B]
            int_r_t=int_r_t.unsqueeze(0),   # [T, B]
            policy_index=policy_index.unsqueeze(0), # [T, B]
            ext_hidden_s=ext_hidden_s,
            int_hidden_s=int_hidden_s,
        )
    
    def _put_unroll_onto_queue(self, unrolled_transition):
        """
        Important note, store hidden states for every step in the unroll will
            consume HUGE memory.
        """
        self._queue.put(unrolled_transition)
    
    def _sample_policy(self):
        """
        Sample new policy from meta collector.
        """
        self._policy_index = self._meta_coll.sample()
        self._policy_beta = self._betas[self._policy_index]
        self._policy_discount = self._gammas[self._policy_index]

    def _update_actor_network(
        self, update_embed: bool = False
    ):
        q_state_dict = self._shared_params['network']
        embed_state_dict= self._shared_params['embedding_network']
        rnd_state_dict = self._shared_params['rnd_predictor_network']

        if update_embed:
            state_net_paris = zip(
                (q_state_dict, rnd_state_dict),
                (self._network, self._rnd_predictor_network),
            )
        
        for state_dict, network in state_net_paris:
            if state_dict is not None:
                if self._device != 'cpu':
                    state_dict = {
                        k: v.to(device=self._device) for k, v in state_dict.items()
                    }
                network.load_state_dict(state_dict)

    @property
    def intrinsic_reward(self) -> float:
        """
        Returns intrinsic reward for last state s_tm1.
        """
        # Equation 1 of the NGU paper.
        return self._episodic_bonus_t \
                * min(max(self._lifelong_bonus_t, 1.0), 5.0)
    
    @property
    def statistics(self) -> Mapping[Text, float]:
        """
        Returns current actor's statistics as a dictionary.
        """
        return {
            # 'policy_index': self._policy_index,
            'policy_discount': self._policy_discount,
            'policy_beta': self._policy_beta,
            'exploration_epsilon': self._exploration_epsilon,
            'intrinsic_reward': self.intrinsic_reward,
            # 'episodic_bonus': self._episodic_bonus_t,
            # 'lifelong_bonus': self._lifelong_bonus_t,
        }


class Learner(types_lib.Learner):
    """
    Agent57 learner.
    """
    
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        embedding_network: nn.Module,
        rnd_target_network: nn.Module,
        rnd_predictor_network: nn.Module,
        intrinsic_optimizer: torch.optim.Optimizer,
        replay: replay_lib.PrioritizedReplay,
        target_net_update_interval: int,
        min_replay_size: int,
        batch_size: int,
        unroll_length: int,
        burn_in: int,
        retrace_lambda: float,
        transformed_retrace: bool,
        priority_eta: float,
        clip_reward: bool,
        max_grad_norm: float,
        device: torch.device,
        shared_params: dict,
    ) -> None:
        """
        Args:
            network: the Q network which want to train and optimize.
            optimizer: the optimizer for Q network.
            embedding_network: NGU action prediction network.
            rnd_target_network: RND random network.
            rnd_predictor_network: RND predictor network.
            intrinsic_optimizer: the optimizer for action prediction and
                RND predictor networks.
            replay: prioritized recurrent experience replay.
            target_net_update_ineterval: how often to copy online network 
                parameters to target.
            min_replay_size: wait till experience replay buffer this number 
                before start to learn.
            batch_size: sample batch_size of transitions.
            burn_in: burn n transitions to generate initial hidden state before
                learning.
            unroll_length: transition sequence length.
            retrace_lambda: coefficient of the retrace lambda.
            transformed_retrace: if True, use transformed retrace.
            priority_eta: coefficient to mix the max and mean absoulte TD errors.
            clip_grad: if True, clip gradients norm.
            max_grad_nrom: the maximum gradient norm for clip grad, only works if 
                clip_grad is True.
            device: pyTorch runtime device.
            shared_params: a shared dict, so we can later update the parameters for 
                actors.
        """
        if not 1 <= target_net_update_interval:
            raise ValueError(
                f'Expect target_net_update_interval to be positive integer, got {target_net_update_interval}'
            )
        if not 1 <= min_replay_size:
            raise ValueError(
                f'Expect min_replay_size to be integer greater than or equal to 1, got {min_replay_size}'
            )
        if not 1 <= batch_size <= 128:
            raise ValueError(
                f'Expect batch_size to in the range [1, 128], got {batch_size}'
            )
        if not 1 <= unroll_length:
            raise ValueError(
                f'Expect unroll_length to be greater than or equal to 1, got {unroll_length}'
            )
        if not 0 <= burn_in < unroll_length:
            raise ValueError(
                f'Expect burn_in length to be [0, {unroll_length}), got {burn_in}'
            )
        if not 0.0 <= retrace_lambda <= 1.0:
            raise ValueError(
                f'Expect retrace_lambda to in the range [0.0, 1.0], got {retrace_lambda}'
            )
        if not 0.0 <= priority_eta <= 1.0:
            raise ValueError(
                f'Expect priority_eta to in the range [0.0, 1.0], got {priority_eta}'
            )
        
        self.agent_name = 'Agent57-learner'
        self._device = device
        self._netowrk = network.to(device=device)
        self._netowrk.train()
        self._optimizer = optimizer
        self._embedding_network = embedding_network.to(device=device)
        self._embedding_network.train()
        self._rnd_predictor_network = rnd_predictor_network.to(
            device=device
        )
        self._rnd_predictor_network.train()
        self._intrinsic_optimizer = intrinsic_optimizer

        self._rnd_target_network = rnd_target_network.to(
            device=device
        )
        # Lazy way to create target Q networks
        self._target_network = copy.deepcopy(
            self._netowrk
        ).to(device=device)

        # Disable autograd for target Q networks, and RND target networks.
        no_autograd(self._target_network)
        no_autograd(self._rnd_target_network)

        self._shared_params = shared_params

        self._batch_size = batch_size
        self._burn_in = burn_in
        self._unroll_length = unroll_length
        self._total_unroll_length = unroll_length + 1
        self._target_net_update_interval = target_net_update_interval
        self._clip_grad = clip_reward
        self._max_grad_norm = max_grad_norm

        # Accumulate running statistics to calculate mean and std
        self._rnd_obs_normalizer = normalizer.TorchRnningMeanStd(
            shape=(1, 84, 84), device=self._device
        )

        self._replay = replay
        self._min_replay_size = min_replay_size
        self._priority_eta = priority_eta

        self._max_see_priority = 1.0    # New unroll will use this as priority

        self._retrace_lambda = retrace_lambda
        self._transformed_retrace = transformed_retrace

        self._step_t = -1
        self._update_t = 0
        self._target_update_t = 0
        self._retrace_loss_t = np.nan
        self._embed_loss_t = np.nan
    
    def step(self) -> Iterable[Mapping[Text, float]]:
        """
        Increment learner step, and potentially do a update when called.

        Yields:
            learner statistics if network parameters update occurred,
                otherwise returns None.
        """
        self._step_t += 1

        if self._replay.size < self._min_replay_size \
            or self._step_t % max(4, int(self._batch_size * 0.25)) != 0:
            return
        
        self._learn()
        yield self.statistics
    
    def reset(self)-> None:
        """
        Should be called at the beginning of every iteration.
        """
    
    def received_item_from_queue(self, item)-> None:
        """
        Received item send by actors through multiprocessing queue.
        """
        self._replay.add(
            item, self._max_see_priority
        )
    
    def get_network_state_dict(self):
        """
        To keep things consistent, move the parameters to CPU.
        """
        return {
            k: v.cpu() for k, v in self._netowrk.state_dict().items()
        }
    
    def get_embedding_network_state_dict(self):
        """
        To keep things consistent, move the parameters to CPU.
        """
        return {
            k: v.cpu() for k, v in self._embedding_network.state_dict().items()
        }
    
    def get_rnd_predictor_network_state_dict(self):
        """
        To keep things consistent, move the parameters to CPU.
        """
        return {
            k: v.cpu() for k, v in self._rnd_predictor_network.state_dict().items()
        }
    
    def _learn(self)-> None:
        transitions, indices, weights = self._replay.sample(self._batch_size)
        priorities = self._update_q_network(transitions, weights)
        self._update_embed_and_rnd_predictor_networks(
            transitions, weights
        )

        self._update_t += 1

        if priorities.shape != (self._batch_size):
            raise RuntimeError(
                f'Expect priorities has shape ({self._batch_size},), '
                f'got {priorities.shape}'
            )
        priorities = np.abs(priorities)
        self._max_seen_priority = np.max(
            [self._max_seen_priority, np.max(priorities)]
        )
        self._replay.update_priorities(indices, priorities)

        self._shared_params['network'] = self.get_network_state_dict()
        self._shared_params['embedding_network'] = self.get_embedding_network_state_dict()
        self._shared_params['rnd_predictor_network'] = \
            self.get_rnd_predictor_network_state_dict()
        
        # Copy Q network parameters to taget Q network, every m updates
        if self._update_t > 1 and self._update_t % self._target_net_update_interval == 0:
            self._update_target_network()
    
    def _update_q_network(
        self,
        transitions: Agent57Transition,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights = torch.from_numpy(weights).to(
            device=self._device,
            dtype=torch.float32,
        )   # [B,]
        base.assert_rank_and_dtype(weights, 1, torch.float32)

        # Get initial hidden state for batch extrinsic and intrinsic Q networks,
        # handle possible burn in.
        init_ext_hidden_s, init_int_hidden_s = \
            self._extract_first_step_hidden_state(transitions)
        burn_transitions, learn_transitions = \
            replay_lib.split_structure(
                transitions, TransitionStructure, self._burn_in
            )
        
        if burn_transitions is not None:
            # Burn in for extrinsic and intrinsic Q networks.
            init_ext_hidden_s, int_hidden_s, target_ext_hidden_s, target_int_hidden_s = \
                self._burn_in_unroll_q_networks(
                    burn_transitions,
                    self._netowrk,
                    self._target_network,
                    init_ext_hidden_s,
                    init_int_hidden_s,
                )
        else:
            # Make copy of hidden state for extrinsic Q networks.
            ext_hidden_s = tuple(
                s.clone().to(device=self._device) for s in init_ext_hidden_s
            )
            target_ext_hidden_s = tuple(
                s.clone().to(device=self._device) for s in init_ext_hidden_s
            )
            # Make copy of hidden state for intrinsic Q networks.
            int_hidden_s = tuple(
                s.clone().to(device=self._device) for s in init_int_hidden_s
            )
            target_int_hidden_s = tuple(
                s.clone().to(device=self._device) for s in init_int_hidden_s
            )
        
        # Update Q network.
        self._optimizer.zero_grad()

        # Do network pass for all four Q networks to get estimated q values.
        ext_q_t, int_q_t = self._get_predicted_q_values(
            learn_transitions, self._netowrk, ext_hidden_s, int_hidden_s
        )

        with torch.no_grad():
            target_ext_q_t, target_int_q_t = self._get_predicted_q_values(
                learn_transitions,
                self._target_network,
                target_ext_hidden_s,
                target_int_hidden_s,
            )
        
        ext_retrace_loss, ext_priorities = self._calc_retrace_loss(
            learn_transitions, ext_q_t, target_ext_q_t.detach()
        )
        int_retrace_loss, int_priorities = self._calc_retrace_loss(
            learn_transitions, int_q_t, target_int_q_t.detach()
        )

        # Multiply loss bt sampling weights, averaging over batch dimension
        loss = torch.mean(
            (ext_retrace_loss + int_retrace_loss) * weights.detach()
        )

        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._netowrk.parameters(),
                self._max_grad_norm,
            )
        
        self._optimizer.step()

        priorities = 0.8 * ext_priorities + 0.2 * int_priorities

        # For logging only.
        self._retrace_loss_t = loss.detach().cpu().item()

        return priorities
    
    def _get_predicted_q_values(
        self,
        transitions: Agent57Transition,
        network: torch.nn.Module,
        ext_hidden_state: HiddenState,
        int_hidden_state: HiddenState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the predicted q values from the network for a given batch 
            of sampled unrolls.

        Args:
            transitions: sampled batch of unrolls, this should not include
                the burn_in part.
            network: this could be any one of the extrinsic and intrinsic 
                (online or target) networks.
            ext_hidden_state: initial hidden states for the network.
            int_hidden_state: initial hidden states for the network.
        """
        s_t = torch.from_numpy(
            transitions.s_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B, state_shape]
        last_action = torch.from_numpy(
            transitions.last_action
        ).to(device=self._device, dtype=torch.int64)    # [T+1, B]
        ext_r_t = torch.from_numpy(
            transitions.ext_r_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        int_r_t = torch.from_numpy(
            transitions.int_r_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        policy_index = torch.from_numpy(
            transitions.policy_index
        ).to(device=self._device, dtype=torch.int64)    # [T+1, B]

        # Rank and dtype checks, note we have a new unroll time dimension, 
        # states may be images, which is rank 5.
        base.assert_rank_and_dtype(
            s_t, (3, 5), torch.float32
        )
        base.assert_rank_and_dtype(
            last_action, 2, torch.long
        )
        base.assert_rank_and_dtype(
            ext_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            int_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            policy_index, 2, torch.long
        )

        # Rank and dtype checks for hidden state.
        base.assert_rank_and_dtype(
            ext_hidden_state[0], 3, torch.float32
        )
        base.assert_rank_and_dtype(
            ext_hidden_state[1], 3, torch.float32
        )
        base.assert_batch_dimension(
            ext_hidden_state[0], self._batch_size, 1
        )
        base.assert_batch_dimension(
            ext_hidden_state[1], self._batch_size, 1
        )
        base.assert_rank_and_dtype(
            int_hidden_state[0], 3, torch.float32
        )
        base.assert_rank_and_dtype(
            int_hidden_state[1], 3, torch.float32
        )
        base.assert_batch_dimension(
            int_hidden_state[0], self._batch_size, 1
        )
        base.assert_batch_dimension(
            int_hidden_state[1], self._batch_size, 1
        )

        # Get q values from Q network,
        output = network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_state=ext_hidden_state,
                int_hidden_state=int_hidden_state,
            )
        )

        return (
            output.ext_q_values, output.int_q_values
        )
    
    def _calc_retrace_loss(
        self,
        transitions: Agent57Transition,
        q_t: torch.Tensor,
        target_q_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        a_t = torch.from_numpy(
            transitions.a_t
        ).to(device=self._device, dtype=torch.int64)    # [T+1, B]
        behavior_prob_a_t = torch.from_numpy(
            transitions.prob_a_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        ext_r_t = torch.from_numpy(
            transitions.ext_r_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        int_r_t = torch.from_numpy(
            transitions.int_r_t
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        beta = torch.from_numpy(
            transitions.beta
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        discount = torch.from_numpy(
            transitions.discount
        ).to(device=self._device, dtype=torch.float32)  # [T+1, B]
        done = torch.from_numpy(
            transitions.done
        ).to(device=self._device, dtype=torch.bool) # [T+1, B]

        # Rank and dtype checks, note have a new unroll time dimension,
        # states may be images, which is rank 5.
        base.assert_rank_and_dtype(
            behavior_prob_a_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            a_t, 2, torch.long
        )
        base.assert_rank_and_dtype(
            ext_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            int_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            beta, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            discount, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            done, 2, torch.bool
        )

        r_t = ext_r_t + beta * int_r_t  # Augmented rewards
        discount_t = (~done).float() * discount # (T+1, B)

        # Derive target policy probabilities from q values.
        target_policy_probs = F.softmax(
            target_q_t, dim=-1
        )   # [T+1, B, action_dim]

        if self._transformed_retrace:
            transform_tx_pair = nonlinear_bellman.SIGNED_HYPERBOLIC_PAIR
        else:
            transform_tx_pair = nonlinear_bellman.IDENTITY_PAIR # No transform
        
        # Compute retrace loss.
        retrate_out = nonlinear_bellman.transformed_retrace(
            q_tm1=q_t[:-1],
            q_t=target_q_t[1:],
            a_tm1=a_t[:-1],
            a_t=a_t[1:],
            r_t=r_t[1:],
            discount_t=discount_t[1:],
            pi_t=target_policy_probs[1:],
            mu_t=behavior_prob_a_t[1:],
            lambda_=self._retrace_lambda,
            tx_pair=transform_tx_pair,
        )

        # Compute priority.
        priorities = distributed.calculate_dist_priorities_from_td_error(
            retrate_out.extra.td_error, self._priority_eta
        )

        # Sums over time dimension.
        loss = torch.sum(retrate_out, dim=0)

        return loss, priorities
    
    def _update_embed_and_rnd_predictor_networks(
        self,
        transitions: Agent57Transition,
        weights: np.ndarray,
    ) -> None:
        """
        Use last 5 frames to update the embedding and RND predictor networks.
        """
        b = self._batch_size
        weights = torch.from_numpy(
            weights[-b:]
        ).to(device=self._device, dtype=torch.float32)  # [B]
        base.assert_rank_and_dtype(
            weights, 1, torch.float32
        )

        self._intrinsic_optimizer.zero_grad()
        # [batch_size]
        rnd_loss = self._calc_rnd_loss(transitions)
        embed_loss = self._calc_embed_inverse_loss(transitions)

        # Multiply loss by sampling weights, averaging over batch dimension
        loss = torch.mean(
            (rnd_loss + embed_loss) * weights.detach()
        )

        loss.backward()

        if self._clip_grad:
            torch.nn.utils.clip_grad_norm_(
                self._rnd_predictor_network.parameters(), self._max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self._embedding_network.parameters(), self._max_grad_norm
            )
        
        self._intrinsic_optimizer.step()

        # For logging only.
        self._rnd_loss_t = rnd_loss.detach().mean().cpu().item()
        self._embed_loss_t = embed_loss.detach().mean().cpu().item()

    def _calc_rnd_loss(
        self,
        transitions: Agent57Transition,
    ) -> torch.Tensor:
        s_t = torch.from_numpy(
            transitions.s_t[-5:]
        ).to(device=self._device, dtype=torch.float32)  # [5, B, state_shape]
        # Rank and dtype checks.
        base.assert_rank_and_dtype(
            s_t, (3, 5), torch.float32
        )
        # Merge batch and time dimension.
        s_t = torch.flatten(s_t, 0, 1)

        normed_s_t = self._normalize_rnd_obs(s_t)

        pred_s_t = self._rnd_predictor_network(normed_s_t)
        with torch.no_grad():
            target_s_t = self._rnd_target_network(normed_s_t)
        
        rnd_loss = torch.square(
            pred_s_t - target_s_t
        ).mean(dim=1)
        # Reshape loss into [5, B].
        rnd_loss = rnd_loss.view(5, -1)

        # Sums over time dimension. shape [B]
        loss = torch.sum(rnd_loss, dim=0)

        return loss
    
    def _calc_embed_inverse_loss(
        self,
        transitions: Agent57Transition,
    ) -> torch.Tensor:
        s_t = torch.from_numpy(
            transitions.s_t[-6:]
        ).to(device=self._device, dtype=torch.float32)  # [6, B, state_shape].
        a_t = torch.from_numpy(
            transitions.a_t[-6:]
        ).to(device=self._device, dtype=torch.int64)    # [6, B]

        # Rank and dtype checks.
        base.assert_rank_and_dtype(
            s_t, (3, 5), torch.float32
        )
        base.assert_rank_and_dtype(
            a_t, 2, torch.long
        )

        s_tm1 = s_t[0:-1, ...]  # [5, B, state_shape]
        s_t = s_t[1:, ...]  # [5, B, state_shape]
        a_tm1 = a_t[:-1, ...]   # [5, B]

        # Merge batch and time dimension.
        s_tm1 = torch.flatten(s_tm1, 0, 1)
        s_t = torch.flatten(s_t, 0, 1)
        a_tm1 = torch.flatten(a_tm1, 0, 1)

        # Compute action prediction loss.
        embedding_s_tm1 = self._embedding_network(s_tm1)    # [5*B, latent_dim]
        embedding_s_t = self._embedding_network(s_t)    # [5*B, latent_dim]
        embeddings = torch.cat(
            [embedding_s_tm1, embedding_s_t], dim=-1
        )
        pi_logits = self._embedding_network.inverse_prediction(embeddings)  # [5*B, action_dim]

        loss = F.cross_entropy(pi_logits, a_tm1, reduction='none')  # [5*B,]
        # Reshape loss into [5, B].
        loss = loss.view(5, -1)

        # Sums over time dimension. shape [B]
        loss = torch.sum(loss, dim=0)
        return loss
    
    @torch.no_grad()
    def _normalize_rnd_obs(self, rnd_obs):
        rnd_obs = rnd_obs.to(
            device=self._device, dtype=torch.float32
        )

        normed_obs = self._rnd_obs_normalizer.normalize(rnd_obs)
        normed_obs = normed_obs.clamp(-5, 5)

        self._rnd_obs_normalizer.update(rnd_obs)

        return normed_obs
    
    @torch.no_grad()
    def _burn_in_unroll_q_networks(
        self,
        transitions: Agent57Transition,
        network: torch.nn.Module,
        target_network: torch.nn.Module,
        ext_hidden_state: HiddenState,
        int_hidden_state: HiddenState,
    ) -> Tuple[
        HiddenState, HiddenState, HiddenState, HiddenState
    ]:
        """
        Unroll both online and target q networks to generate hidden states
            for LSTM.
        """
        s_t = torch.from_numpy(
            transitions.s_t
        ).to(device=self._device, dtype=torch.float32)  # [burn_in, B, state_shape]
        last_action = torch.from_numpy(
            transitions.last_action
        ).to(device=self._device, dtype=torch.int64)    # [burn_in, B]
        ext_r_t = torch.from_numpy(
            transitions.ext_r_t
        ).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        int_r_t = torch.from_numpy(
            transitions.int_r_t
        ).to(device=self._device, dtype=torch.float32)  # [burn_in, B]
        policy_index = torch.from_numpy(
            transitions.policy_index
        ).to(device=self._device, dtype=torch.int64)    # [burn_in, B]

        # Rank and dtype checs, note have a new unroll time dimension, states may be images,   
        # which is rank 5.
        base.assert_rank_and_dtype(
            s_t, (3, 5), torch.float32
        )
        base.assert_rank_and_dtype(
            last_action, 2, torch.long
        )
        base.assert_rank_and_dtype(
            ext_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            int_r_t, 2, torch.float32
        )
        base.assert_rank_and_dtype(
            policy_index, 2, torch.long
        )

        _ext_hidden_s = tuple(
            s.clone().to(device=self._device) for s in ext_hidden_state
        )
        _int_hidden_s = tuple(
            s.clone().to(device=self._device) for s in int_hidden_state
        )
        _target_ext_hidden_s = tuple(
            s.clone().to(device=self._device) for s in ext_hidden_state
        )
        _target_int_hidden_s = tuple(
            s.clone().to(device=self._device) for s in int_hidden_state
        )

        # Burn in to generate hidden state for LSTM, unroll both online and target
        # Q networks
        output = network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_s=_ext_hidden_s,
                int_hidden_s=_int_hidden_s,
            )
        )

        target_output = target_network(
            Agent57NetworkInputs(
                s_t=s_t,
                a_tm1=last_action,
                ext_r_t=ext_r_t,
                int_r_t=int_r_t,
                policy_index=policy_index,
                ext_hidden_s=_target_ext_hidden_s,
                int_hidden_s=_target_int_hidden_s,
            )
        )

        return (
            output.ext_hidden_s,
            output.int_hidden_s,
            target_output.ext_hidden_s,
            target_output.int_hidden_s,
        )
    
    def _extract_first_step_hidden_state(
        self,
        transitions: Agent57Transition,
    ) -> Tuple[HiddenState, HiddenState]:
        """
        Returns ext_hidden_state and int_hidden_state.
        """
        # Only need the first step hidden states in replay, shape 
        # [batch_sze, num_lstm_layers, lstm_hidden_size].
        ext_init_h = torch.from_numpy(
            transitions.ext_init_h[0:1]
        ).squeeze(0).to(device=self._device, dtype=torch.float32)
        ext_init_c = torch.from_numpy(
            transitions.ext_init_c[0:1]
        ).squeeze(0).to(device=self._device, dtype=torch.float32)
        int_init_h = torch.from_numpy(
            transitions.int_init_h[0:1]
        ).squeeze(0).to(device=self._device, dtype=torch.float32)
        int_init_c = torch.from_numpy(
            transitions.int_init_c[0:1]
        ).squeeze(0).to(device=self._device, dtype=torch.float32)

        # Rank and dtype checks.
        base.assert_rank_and_dtype(
            ext_init_h, 3, torch.float32
        )
        base.assert_rank_and_dtype(
            ext_init_c, 3, torch.float32
        )
        base.assert_rank_and_dtype(
            int_init_h, 3, torch.float32
        )
        base.assert_rank_and_dtype(
            int_init_c, 3, torch.float32
        )

        # Swap batch and num_lstm_layers axis.
        ext_init_h = ext_init_h.swapaxes(0, 1)
        ext_init_c = ext_init_c.swapaxes(0, 1)
        int_init_h = int_init_h.swapaxes(0, 1)
        int_init_c = int_init_c.swapaxes(0, 1)

        # Batch dimension checks.
        base.assert_batch_dimension(
            ext_init_h, self._batch_size, 1
        )
        base.assert_batch_dimension(
            ext_init_c, self._batch_size, 1
        )
        base.assert_batch_dimension(
            int_init_h, self._batch_size, 1
        )
        base.assert_batch_dimension(
            int_init_c, self._batch_size, 1
        )

        return (
            (ext_init_h, ext_init_c), (int_init_h, int_init_c)
        )
    
    def _update_target_network(self):
        self._target_network.load_state_dict(
            self._netowrk.state_dict()
        )
        self._target_update_t += 1
    
    @property
    def statistics(self)-> Mapping[Text, float]:
        """
        Returns current agent statistics as a dictionary.
        """
        return {
            # 'ext_lr': self._optimizer.param_groups[0]['lr'],
            # 'int_lr': self._intrinsic_optimizer.param_groups[0]['lr']
            'retrace_loss': self._retrace_loss_t,
            'embed_loss': self._embed_loss_t,
            'rnd_loss': self._rnd_loss_t,
            'updates': self._update_t,
            'target_updates': self._target_update_t,
        }