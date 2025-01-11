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


def double_qlearning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t_value: torch.Tensor,
    q_t_selector: torch.Tensor,
)-> base.LossOutput:
    r"""
    Implements the doulbe Q-learning loss.

    The loss is 0.5 times the squared difference between 'q_tm1[a_tm1]' and
        the target 'r_t + discount_t * q_t_value[argmax q_t_selector]'.
    
    See "Double Q-learning" by van Hasselt.
        (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

    Args:
        q_tm1: Tensor holding Q-values for first timestep in a batch of
            transitions, shape [B x action_dim].
        a_tm1: Tensor holding action indices, shape [B].
        r_t: Tensor holding rewards, shape [B].
        discount_t: Tensor holding discount values, shape [B].
        q_t_value: Tensor Q-values for second timestep in a batch of transitions,
            used to estimate the value of the best action, shape [B x action_dim].
        q_t_selector: Tensor of Q-values for second timestep in a batch of
            transitions used to estimate the best action, shape [B x action_dim].

    Returns:
        A namedtuple with fields:
            * `loss`: a tensor containing the batch of losses, shape [B].
            * `extra`: a namedtuple with fields:
                * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
                * `td_error`: batch of temporal difference errors, shape [B].
                * `best_action`: batch of greedy actions wrt `q_t_selector`, shape [B].
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
        q_t_value, 2, torch.float32
    )
    base.assert_rank_and_dtype(
        q_t_selector, 2, torch.float32
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
        q_t_value, q_tm1.shape[0]
    )
    base.assert_batch_dimension(
        q_t_selector, q_tm1.shape[0]
    )

    # double Q-learning op.
    # Build target and select head to update.

    best_action = torch.argmax(
        q_t_selector, dim=1
    )
    # B = q_tm1.shape[0]
    # double_q_bootstrapped = q_t_value[torch.arange(0, B), best_action]
    double_q_bootstrapped = base.batched_index(
        q_t_value, best_action
    )

    with torch.no_grad():
        target_tm1 = r_t + discount_t * double_q_bootstrapped
    
    # qa_tm1 = q_tm1[torch.arange(0, B), a_tm1]
    qa_tm1 = base.batched_index(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target_tm1 - qa_tm1
    loss = 0.5 * td_error**2

    return base.LossOutput(
        loss, DoubleQExtra(
            target_tm1, td_error, best_action
        )
    )


def _slice_with_actions(
    embeddings: torch.Tensor,
    actions: torch.Tensor,
)-> torch.Tensor:
    """
    Slice a Tensor.

    Take embeddings of the form [batch_size, action_dim, embed_dim]
        and actions of the form [batch_size, 1], and return the sliced embeddings
        like embeddings[:, actions, :].

    Args:
        embeddings: Tensor of embeddings to index.
        actions: int Tensor to use as index into embeddings.

    Returns:
        Tensor of embeddings indexed by actions.
    """

    batch_size, action_dim = embeddings.shape[:2]

    # Values are the 'values' in a sparse tensor we will be setting
    act_idx = actions[:, None]

    values = torch.reshape(
        torch.ones(
            actions.shape,
            dtype=torch.int8,
            device=actions.device,
        ),
        [-1],
    )

    # Create a range for each index into the batch
    act_range = torch.arange(
        0, batch_size, dtype=torch.int64
    )[:, None].to(
        device=actions.device
    )
    # Combine this into coordings with the action indices
    indices = torch.concat(
        [act_range, act_idx], 1
    )

    # Needs transpose indices before adding to torch.sparse_coo_tensor.
    actions_mask = torch.sparse_coo_tensor(
        indices.t(),
        values,
        [batch_size, action_dim]
    )
    with torch.no_grad():
        actions_mask = actions_mask.to_dense().bool()
    
    sliced_emb = torch.masked_select(
        embeddings, actions_mask[..., None]
    )
    # Make sure shape is the same as embeddings
    sliced_emb = sliced_emb.reshape(embeddings.shape[0], -1)

    return sliced_emb


def l2_project(
    z_p: torch.Tensor,
    p: torch.Tensor,
    z_q: torch.Tensor,
)-> torch.Tensor:
    r"""
    Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.

    The supports z_p and z_q are speficied as tensors of distinct atoms (given
        in ascending order).

    Let kq be len(z_q) and kp be len(z_p). This projection works for any
        support z_q, in particular kq need not be equal to kp.

    Args:
        z_p: Tensor holding support of distribution p, shape '[batch_size, kp]'.
        p: Tensor holding probability values p(z_p[i]), shape '[batch_size, kp]'.
        z_q: Tensor holding support to project onto, shape '[kq]'.

    Returns:
        Projection of (z_p, p) onto support z_q under Cramer distance.
    """
    