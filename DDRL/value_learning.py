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


class DoubleQExtra(NamedTuple):
    target: torch.Tensor
    td_error: torch.Tensor
    best_action: torch.Tensor


class Extra(NamedTuple):
    target: Optional[torch.Tensor]


def qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t: torch.Tensor,
)-> base.LossOutput:
    r"""
    Implements the Q-learning loss.

    The loss is '0.5' times the squared difference between 'q_tmq[a_tm1]' and
        the target 'r_t + discount_t * max_q_t'.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
        (http://incompleteideas.net/book/ebook/node65.html).

    Args:
        q_tm1: Tensor holding Q-values for first timestepp in a batch of
            transitions, shape [B x action_dim].
        a_tm1: Tensor holding action indicies, shape '[B]'.
        r_t: Tensor holding rewards, shape '[B]'.
        discount_t; Tensor holding discount values, shape '[B]'.
        q_t: Tensor holding Q-values for secend timestep in a batch of 
            transitions, shape '[B x action_dim]'.

    Returns:
        A namedtuple with fields:
            * `loss`: a tensor containing the batch of losses, shape `[B]`.
            * `extra`: a namedtuple with fields:
                * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
                * `td_error`: batch of temporal difference errors, shape `[B]`.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(
        q_tm1, 2, torch.float32
    )
    base.assert_rank_and_dtype(
        a_tm1, 1, torch.long
    )
    base.assert_rank_and_dtype(
        r_t, 1, torch.float32
    )
    base.assert_rank_and_dtype(
        discount_t, 1, torch.float32
    )
    base.assert_rank_and_dtype(
        q_t, 2, torch.float32
    )

    base.assert_batch_dimension(
        a_tm1, q_tm1.shape[0]
    )
    base.assert_batch_dimension(
        r_t, q_tm1.shape[0]
    )
    base.assert_batch_dimension(
        discount_t, q_tm1.shape[0]
    )
    base.assert_batch_dimension(
        q_t, q_tm1.shape[0]
    )

    # Q-learning op.
    # Build target and select head to update.
    with torch.no_grad():
        target_tm1 = r_t + discount_t * torch.max(q_t, dim=1)[0]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)
    # B = q_tm1.shape[0]
    # qa_tmq = q_tm1[torch.arange(0, B), a_tm1]

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(
        loss, QExtra(target_tm1, td_error)
    )


