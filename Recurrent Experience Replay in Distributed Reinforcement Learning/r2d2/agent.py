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

