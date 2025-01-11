"""
Transformed value functions.

Canonical value functions map states onto the expected discounted sum of rewards
    that may be collected by an agent from any starting state. Value functions may
    also be defined as the fixed points of certain linear recursive relations known
    as Bellman equations. It is sometimes useful lto consider transformed values that
    are the solution to non-linear generalization of traditional Bellman equations.

In this subpackage provide a general utility for wrapping bootstrapped return
    calculations to construct regression targets for these transformed values.
    Also use this to implement different learning algorithms from the literature.
"""

from typing import (
    NamedTuple,
    Callable,
    Any,
)
import functools

import torch

from . import (
    base,
    value_learning,
    multistep,
    transforms,
)


class TxPair(NamedTuple):
    apply: callable[[Any], Any]
    apply_inv: Callable[[Any], Any]



# Example transform pairs; these typically consist of a monotonically increasing
# squashing fn `apply` and its inverse `apply_inv`. Ohter choices are possible.

IDENTITY_PAIR = TxPair(
    transforms.identity,
    transforms.identity,
)
SIGNED_LOGP1_PAIR = TxPair(
    transforms.signed_logp1,
    transforms.signed_expm1,
)
SIGNED_HYPERBOLIC_PAIR = TxPair(
    transforms.signed_hyperbolic,
    transforms.signed_parabolic,
)
HYPERBOLIC_SIN_PAIR = TxPair(
    transforms.hyperbolic_arcsin,
    transforms.hyperbolic_sin,
)


def transform_values(
    build_targets, *value_argnums
):
    """
    Decorator to convert targets to use transformed value function.
    """

    @functools.wraps(build_targets)
    def wrapped_build_targets(
        tx_pair, *args, **kwargs
    ):
        tx_args = list(args)
        for index in value_argnums:
            tx_args[index] = tx_pair.apply_inv(tx_args[index])
        
        targets = build_targets(*tx_args, **kwargs)
        return tx_pair.apply(targets)
    
    return wrapped_build_targets()


transformed_general_off_policy_returns_from_action_values = transform_values(
    multistep.general_off_policy_returns_from_action_values, 0
)


def transformed_retrace(
    q_tm1: torch.Tensor,
    q_t: torch.Tensor,
    a_tm1: torch.Tensor,
    a_t: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    pi_t: torch.Tensor,
    mu_t: torch.Tensor,
    lambda_: float,
    eps: float = 1e-8,
    tx_pair: TxPair = IDENTITY_PAIR,
)-> base.LossOutput:
    