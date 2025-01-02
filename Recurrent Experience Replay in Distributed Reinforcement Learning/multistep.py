"""
Common ops for multistep return evaluation.
"""

import torch
import numpy as np

import base



def n_step_bellman_target(
    r_t: torch.Tensor,
    done: torch.Tensor,
    q_t: torch.Tensor,
    gamma: float,
    n_steps: int,
)-> torch.Tensor:
    r"""
    Computes n-step Bellman targets.

    See section 2.3 of R2D2 paper (which does not mention the logic around end of
    episode).

    Args:
        reward: This is r_t in the equations below. Should be non-discounted, non-summed,
            shape [T, B] tensor.
        done: This is done_t in the equations below. done_t should be true
            if the episode is done just after
            experimenting reward r_t, shape [T, B] tensor.
        q_t: This is Q_target(s_{t+1}, a*) (where a* is an action chosen by the caller),
            shape [T, B] tensor.
        gamma: Exponential RL discounting.
        n_steps: The number of steps to loock ahead for computing the Bellman targets.

    Returns:
        y_t targets as <float32>[time, batch_size] tensor.
        When n_steps=1, this is just:
            $$r_t + gamma * (1 - done_t) * Q_{target}(s_{t+1}, a^*)$$
        In the general case, this is:
            $$(\sum_{i=0} \ gamma ^ {i} * notdone_{t, i-1} * r_{t + i}) + 
                \gamma ^ n * notdone_{t, n-1} * Q_{target}(s_{t + n}, a^*) $$
        where notdone_{t,i} is defined as:
            $$notdone_{t,i} = \prod_{k=0}^{k=i}(1 - done_{t+k})$$
        The last n_step-1 targets cannot be computed with n_step returns, since we
            run out of Q_{target}(s_{t+n}). Instead, they will use n_step-1, .., 1 step
            returns. For those last targets, the last Q_{target}(s_{t}, a^*) is re-used
            multiple times.
    """
    # Rank and compatibility checks.
    base.assert_rank_and_dtype(
        r_t, 2, torch.float32
    )
    base.assert_rank_and_dtype(
        done, 2, torch.bool
    )
    base.assert_rank_and_dtype(
        q_t, 2, torch.float32
    )

    base.assert_batch_dimension(
        done, q_t.shape[0]
    )
    base.assert_batch_dimension(
        r_t, q_t.shape[0]
    )
    base.assert_batch_dimension(
        done, q_t.shape[1], 1
    )
    base.assert_batch_dimension(
        r_t, q_t.shape[1], 1
    )

    # We append n_steps - 1 times the last q_target. They are divied by gamma **
    # k to correct for the fact that they are at a 'fake' indices, and will
    # therefore end up being multiplied back by gamma ** k in the loop below.
    # We prepend 0s that will be discarded at the first iteration below.
    bellman_target = torch.concat(
        [torch.zeros_like(q_t[0:1], q_t)] \
        + [q_t[-1:] / gamma**k for k in range(1, n_steps)],
        dim=0,
    )
    # Pad with n_steps 0s. They will be used to compute the last n_steps-1
    # targets (having 0 values is important).
    done = torch.concat(
        [done] + [torch.zeros_like(done[0:1])] * n_steps, dim=0
    )
    rewards = torch.concat(
        [r_t] + [torch.zeros_like(r_t[0:1])] * n_steps, dim=0
    )
    # Iteratively build the n_stps targets. After the i-th iteration (1-based),
    # bellman_target is effectively the i-step returns.
    for _ in range(n_steps):
        rewards = rewards[:-1]
        done = done[:-1]
        bellman_target = rewards + gamma \
                            * (1.0 - done.float()) * bellman_target[1:]
    
    return bellman_target


def truncated_generalized_advantage_estimation(
    r_t: torch.Tensor,
    value_t: torch.Tensor,
    value_tp1: torch.Tensor,
    discount_tp1: torch.Tensor,
    lambda_: float,
)-> torch.Tensor:
    """
    Computes truncated generalized advantage estimates for a sequence length k.

    The advantages are computed in a backwards fashion according to the equation:
        Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
        where δₜ = rₜ + γₜ * v(sₜ₊₁) - v(sₜ).

    See Proximal Policy Optimization Algorithms, Schulman et al.:
        https://arxiv.org/abs/1707.06347.

    Args:
        r_t: sequences of rewards at times [0, k]
        value_t: sequence of values under π at times [0, k]
        value_tp1: seuquence of values under π at times [1, k+1]
        discount_tp1: sequence of discounts at times [1, k+1]
        lambda_: a scalar

    Returns:
        Multistep truncated generalized advantage estimation at times [0, k-1].
    """

    base.assert_rank_and_dtype(
        r_t, 1, torch.float32
    )
    base.assert_rank_and_dtype(
        value_t, 1, torch.float32
    )
    base.assert_rank_and_dtype(
        value_tp1, 1, torch.float32
    )
    base.assert_rank_and_dtype(
        discount_tp1, 1, torch.float32
    )

    lambda_ = torch.ones_like(discount_tp1) * lambda_   # If scalar, make into vector.
    delta_t = r_t + discount_tp1 * value_tp1 - value_t
    advantage_t = torch.zeros_like(
        delta_t, dtype=torch.float32
    )

    gae_t = 0
    for i in reversed(range(len(delta_t))):
        gae_t = delta_t[i] + discount_tp1[i] * lambda_[i] * gae_t
        advantage_t[i] = gae_t
    
    return advantage_t


def general_off_policy_returns_from_action_values(
    q_t: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    c_t: torch.Tensor,
    pi_t: torch.Tensor,
)-> torch.Tensor:
    """
    Calculates targets for various off-policy correction algorithms.

    Given a window of experience of length `k` generated by a behavior policy 
    """