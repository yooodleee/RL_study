"""
R2D2 agent class.

From the parper "Recurrent Experience Replay in Distributed Reinforcement Learning"
https://openreview.net/pdf?id=r1lyTjAqYX.

The code for value function rescaling, inverse value function resacling, and n-step bellman targets are from seed-rl:
https://github.com/google-research/seed_rl/blob/66e8890261f09d0355e8bf5f1c5e41968ca9f02b/agents/r2d2/learner.py

This agent supports store hidden state (only first step in a unroll) in replay, and burn in.
In fact, even if we use burn in, we're still going to store the hidden state (only first step in a unroll) in the replay.
"""
from typing import Mapping, Optional, Tuple, NamedTuple, Iterable, Text
import copy
import multiprocessing
import numpy
import torch
from torch import nn

# pylint: disable=import-error
import deep_rl_zoo.replay as replay_lib
import deep_rl_zoo.types as types_lib
from deep_rl_zoo import (
    base,
    multistep,
    distributed,
    transforms,
)
from deep_rl_zoo.networks.value import RnnDqnNetworkInputs

torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]



class R2d2Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env.step().

    last_action is the last agent the agent took, before in s_t.
    """

    s_t: Optional[numpy.ndarray]
    r_t: Optional[float]
    done: Optional[bool]
    a_t: Optional[int]
    q_t: Optional[numpy.ndarray]    # q values for s_t
    last_action: Optional[int]
    init_h: Optional[numpy.ndarray] # nn.LSTM initial hidden state
    init_c: Optional[numpy.ndarray] # nn.LSTM initial cell state

TransitionStructure = R2d2Transition(
    s_t=None,
    r_t=None,
    done=None,
    a_t=None,
    q_t=None,
    last_action=None,
    init_h=None,
    init_c=None,
)

def no_autograd(net: torch.nn.Module):
    """
    Disable autograd for a network.
    """
    for p in net.parameters():
        p.requires_grad = False


def calculate_losses_and_priorities(
    q_value: torch.Tensor,
    action: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    target_qvalue: torch.Tensor,
    target_action: torch.Tensor,
    gamma: float,
    n_step: int,
    eps: float = 0.001,
    eta: float = 0.9,   
)-> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculate loss and priority for given samples.
    
    T is the unrolled length, B the batch size, N is number of actions.

    Args:
        q_value: (T+1, B, action_dim) the predicted q values for a given state 's_t' from online Q network.
        action: [T+1, B] the actual action the agent take in state 's_t'.
        reward: [T+1, B] the reward the agent received at timestep t, this is for (s_tm1, a_tm1).
        done: [T+1, B] terminal mask for timestep t, state 's_t'.
        target_qvalue: (T+1, B, N) the estimated TD n-step target values from target Q network,
            this could also be the same q values when just calculate priorities to insert into replay.
        target_action: [T+1, B] the best action to take in t+n timestep target state.
        gamma: discount rate.
        n_step: TD n-step size.
        eps: constant for value function rescaling and inverse function rescaling.
        eta: constant for calculate mixture priorities.

    Returns:
        losses: the losses for given unrolled samples, shape (B, )
        priorities: the priority for given samples, shape (B, )
    """

    base.assert_rank_and_dtype(q_value, 3, torch.float32)
    base.assert_rank_and_dtype(target_qvalue, 3, torch.float32)
    base.assert_rank_and_dtype(reward, 2, torch.float32)
    base.assert_rank_and_dtype(action, 2, torch.long)
    base.assert_rank_and_dtype(target_action, 2, torch.long)
    base.assert_rank_and_dtype(done, 2, torch.bool)

    q_value = q_value.gather(-1, action[..., None]).squeeze(-1) # [T, B]

    target_q_max = target_qvalue.gather(-1, target_action[..., None]).squeeze(-1)   # [T, B]
    # Apply invertible value rescaling to TD target.
    target_q_max = transforms.signed_parabolic(target_q_max, eps)

    # Note the input rewards into 'n_step_bellman_target' should be non-discounted, non-summed.
    target_q = multistep.n_step_bellman_target(
        r_t=reward,
        done=done,
        q_t=target_q_max,
        gamma=gamma,
        n_steps=n_step
    )

    # q_value is actually Q(s_t, a_t), but rewards is for 's_tm1', 'a_tm1',
    # that means our 'target_q' value is one step behind 'q_value'
    # so we need to shift them to make it in the same timestep.
    q_value = q_value[:-1, ...]
    target_q = target_q[1:, ...]

    # Apply value rescaling to TD target.
    target_q = transforms.signed_hyperbolic(target_q, eps)

    if q_value.shape != target_q.shape:
        raise RuntimeError(
            f'Expect q_value and target_q have the same shape, got {q_value.shape} and '
            f'{target_q.shape}'
        )
    
    td_error = target_q - q_value

    with torch.no_grad():
        priorities = distributed.calculate_dist_priorities_from_td_error(
            td_error, eta
        )
    
    # Sums over time dimension.
    losses = 0.5 * torch.sum(torch.square(td_error), dim=0) # [B]

    return losses, priorities


